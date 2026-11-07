// Text-only entrypoint using the checked-in public Engine contract.
#include "ninfer/engine.h"
#include "product/speculative_options.h"
#include <nlohmann/json.hpp>

#include <filesystem>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
struct Task {
    std::string id;
    std::string text;
};

std::uint32_t positive(const std::string& value, const char* option, bool allow_zero = false) {
    std::size_t consumed = 0;
    const auto number = std::stoull(value, &consumed);
    if (consumed != value.size() || value.starts_with('-') || (!allow_zero && number == 0) ||
        number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(option) + " requires a positive uint32");
    }
    return static_cast<std::uint32_t>(number);
}

std::vector<Task> read_tasks(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    if (!file) { throw std::invalid_argument("cannot open tasks: " + path); }
    std::vector<Task> tasks;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') { line.pop_back(); }
        if (line.starts_with("###ID ")) {
            tasks.push_back({line.substr(6), {}});
        } else if (!tasks.empty()) {
            tasks.back().text += line + "\n";
        } else if (!line.empty()) {
            throw std::invalid_argument("tasks must start with ###ID <name> (UTF-8 without BOM)");
        }
    }
    if (tasks.empty()) { throw std::invalid_argument("no tasks found in " + path); }
    return tasks;
}

void usage() {
    std::cout << "ninfer_mini_nordc.exe artifact.ninfer [--prompt TEXT | --tasks FILE]\n"
                 "  --ctx N (default 16384) --max-new N (default 128)\n"
                 "  --kv rk4v4-e8|bf16 --spec none|mtp|dflash2 --draft-tokens N\n"
                 "  --prefill-chunk N --no-graph --thinking on|off\n"
                 "  --temperature N --seed N --max-concurrency 1\n"
                 "  --kvmem --kv-selected N --kv-reserve N --kv-sink N --kv-recent N\n"
                 "  --kv-host-gib N --kv-lrfu-lambda X (KVMem requires eager execution)\n"
                 "  --token-counts (NDJSON tokenizer/template counts; no prefill/generation)\n"
                 "  --token-count-files (persistent counts; UTF-8 task paths on stdin, EOF exits)\n"
                 "One process owns one GPU job. Task files use ###ID <name> boundaries.\n";
}
} // namespace

int run(int argc, char** argv) {
    try {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
        if (argc == 1 || std::string(argv[1]) == "--help") { usage(); return 0; }
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = std::filesystem::u8path(argv[1]);
        engine_options.max_context = 16384;
        engine_options.kv_cache = ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8;
        // Avoid the public API's default 8 GiB host cache on a 32 GiB host.
        engine_options.context_cache.enabled = false;
        engine_options.context_cache.host_state_slots = 0;
        engine_options.context_cache.host_kv_capacity_bytes = 0;
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 128;
        request.execution.sampling.temperature = 0.0F;
        request.execution.sampling.seed = 0;
        bool thinking = false;
        bool token_counts = false;
        bool token_count_files = false;
        std::string prompt;
        std::string task_path;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--no-graph") { engine_options.use_cuda_graph = false; continue; }
            if (option == "--kvmem") { engine_options.kvmem.enabled = true; continue; }
            if (option == "--token-counts") { token_counts = true; continue; }
            if (option == "--token-count-files") {
                token_counts = true;
                token_count_files = true;
                continue;
            }
            if (option == "--help") { usage(); return 0; }
            if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + option); }
            const std::string value = argv[++i];
            if (option == "--prompt") { prompt = value; }
            else if (option == "--tasks") { task_path = value; }
            else if (option == "--ctx") { engine_options.max_context = positive(value, "--ctx"); }
            else if (option == "--max-new") {
                request.execution.requested_output_tokens = positive(value, "--max-new");
            } else if (option == "--prefill-chunk") {
                engine_options.prefill_chunk = positive(value, "--prefill-chunk");
            } else if (option == "--max-concurrency") {
                if (value != "1") { throw std::invalid_argument("HIP harness supports concurrency 1"); }
            } else if (option == "--kv") {
                if (value == "rk4v4-e8") {
                    engine_options.kv_cache = ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8;
                } else if (value == "bf16") { engine_options.kv_cache = ninfer::KvCacheStorage::BFloat16; }
                else { throw std::invalid_argument("unsupported --kv: " + value); }
            } else if (option == "--spec") {
                if (value == "mtp") { engine_options.speculative.backend = ninfer::SpeculativeBackend::Mtp; }
                else if (value == "dflash2") { engine_options.speculative.backend = ninfer::SpeculativeBackend::DFlash2; }
                else if (value == "none") { engine_options.speculative.backend = ninfer::SpeculativeBackend::None; }
                else { throw std::invalid_argument("--spec must be none, mtp or dflash2"); }
            } else if (option == "--draft-tokens") {
                engine_options.speculative.draft_tokens = positive(value, "--draft-tokens");
            } else if (option == "--thinking") {
                if (value != "on" && value != "off") { throw std::invalid_argument("--thinking must be on or off"); }
                thinking = value == "on";
            } else if (option == "--temperature") {
                std::size_t consumed = 0;
                const float number = std::stof(value, &consumed);
                if (consumed != value.size() || !std::isfinite(number) || number < 0) {
                    throw std::invalid_argument("invalid --temperature");
                }
                request.execution.sampling.temperature = number;
            } else if (option == "--seed") {
                std::size_t consumed = 0;
                const auto number = std::stoull(value, &consumed);
                if (consumed != value.size() || value.starts_with('-')) {
                    throw std::invalid_argument("invalid --seed");
                }
                request.execution.sampling.seed = number;
            } else if (option == "--kv-selected") {
                engine_options.kvmem.selected_tokens = positive(value, "--kv-selected");
            } else if (option == "--kv-reserve") {
                engine_options.kvmem.reserve_tokens = positive(value, "--kv-reserve");
            } else if (option == "--kv-sink") {
                engine_options.kvmem.sink_tokens = positive(value, "--kv-sink", true);
            } else if (option == "--kv-recent") {
                engine_options.kvmem.recent_tokens = positive(value, "--kv-recent", true);
            } else if (option == "--kv-host-gib") {
                engine_options.kvmem.host_capacity_bytes =
                    static_cast<std::uint64_t>(positive(value, "--kv-host-gib")) << 30;
            } else if (option == "--kv-lrfu-lambda") {
                std::size_t consumed = 0;
                const double number = std::stod(value, &consumed);
                if (consumed != value.size() || !std::isfinite(number) || number < 0 || number > 1) {
                    throw std::invalid_argument("--kv-lrfu-lambda must be in [0,1]");
                }
                engine_options.kvmem.lrfu_lambda = number;
            } else { throw std::invalid_argument("unknown option: " + option); }
        }
        if (!prompt.empty() && !task_path.empty()) {
            throw std::invalid_argument("choose --prompt or --tasks");
        }
        if (token_count_files && (!prompt.empty() || !task_path.empty())) {
            throw std::invalid_argument("--token-count-files reads task paths from stdin; omit --prompt/--tasks");
        }
        if (!token_count_files && prompt.empty() && task_path.empty()) {
            throw std::invalid_argument("--prompt or --tasks is required");
        }
        ninfer::product::validate_speculative_cli_options(engine_options.speculative);
        if (engine_options.kvmem.enabled || token_counts) { engine_options.use_cuda_graph = false; }
        const std::uint64_t physical_tokens = engine_options.kvmem.enabled
            ? std::uint64_t(engine_options.kvmem.selected_tokens) + engine_options.kvmem.reserve_tokens
            : engine_options.max_context;
        if (physical_tokens > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("physical KV capacity exceeds uint32");
        }
        engine_options.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(physical_tokens));
        if (!std::filesystem::is_regular_file(engine_options.artifact_path)) {
            throw std::invalid_argument("artifact does not exist");
        }
        auto initial_tasks = token_count_files ? std::vector<Task>{}
            : task_path.empty() ? std::vector<Task>{{"prompt", prompt}} : read_tasks(task_path);
        ninfer::Engine engine(std::move(engine_options));
        const auto process_tasks = [&](std::vector<Task>& tasks) {
        for (auto& task : tasks) {
            const auto raw_tokens = token_counts ? engine.tokenize_text(task.text).size() : 0;
            ninfer::PromptInput input;
            input.options.enable_thinking = thinking;
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            ninfer::MessagePart part;
            part.text = std::move(task.text);
            message.parts.push_back(std::move(part));
            input.messages.push_back(std::move(message));
            auto prepared = engine.prepare(std::move(input));
            if (token_counts) {
                std::cout << nlohmann::json{{"id", task.id}, {"raw_tokens", raw_tokens},
                    {"prepared_prompt_tokens", prepared.summary().prompt_tokens}}.dump() << '\n'
                          << std::flush;
                continue;
            }
            const auto prompt_tokens = prepared.summary().prompt_tokens;
            const auto result = engine.generate(std::move(prepared), request);
            std::cout << "###ID " << task.id << '\n';
            if (!result.reasoning.empty()) { std::cout << "<think>" << result.reasoning << "</think>\n"; }
            std::cout << result.content << '\n' << std::flush;
            std::cerr << "[result] generated=" << result.generated_token_ids.size()
                      << " prompt=" << prompt_tokens
                      << " reused=" << result.reused_prompt_tokens
                      << " prefill_seconds=" << result.timings.prefill_seconds
                      << " decode_seconds=" << result.timings.decode_seconds
                      << " mtp_drafted=" << result.speculative.drafted_tokens
                      << " mtp_accepted=" << result.speculative.accepted_tokens << '\n';
        }
        };
        if (token_count_files) {
            std::cout << nlohmann::json{{"type", "ready"}}.dump() << '\n' << std::flush;
            std::string path;
            while (std::getline(std::cin, path)) {
                if (!path.empty() && path.back() == '\r') { path.pop_back(); }
                if (path.empty()) { break; }
                auto tasks = read_tasks(path);
                process_tasks(tasks);
                std::cout << nlohmann::json{{"type", "file_complete"}, {"path", path},
                    {"task_count", tasks.size()}}.dump() << '\n' << std::flush;
            }
        } else {
            process_tasks(initial_tasks);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** wide_argv) {
    std::vector<std::string> arguments;
    arguments.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[i], -1,
                                              nullptr, 0, nullptr, nullptr);
        if (bytes == 0) { std::cerr << "ERROR: invalid Unicode command line\n"; return 1; }
        std::string value(static_cast<std::size_t>(bytes), '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[i], -1,
                            value.data(), bytes, nullptr, nullptr);
        value.pop_back();
        arguments.push_back(std::move(value));
    }
    std::vector<char*> argv;
    for (auto& value : arguments) { argv.push_back(value.data()); }
    return run(argc, argv.data());
}
#else
int main(int argc, char** argv) { return run(argc, argv); }
#endif
