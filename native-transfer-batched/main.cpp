// GENERATED for polymul/negacyclic-batched@1.0.0. Do not edit — `--update` rewrites it.
//
//     ./solution <point directory>
//
// Reads the directory a bundle prepared, answers every case in it, and writes
// what each one cost.
//
//     manifest.json     the point, and how many cases      read
//     cases/000000/     one file per argument              read
//     out/000000/       one file per result                written
//     out/results.json  what each case cost                written
//
// Only the call to fherma_run is timed. Reading and writing are outside the
// window, which is also why seeing the inputs early is not a way to answer
// early: nothing of yours runs during the read. fherma_h2d and fherma_d2h,
// where the answer has them, are clocked apart.
#include "fherma.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

// FHERMA_DEVICE_BEGIN
// What runs when the answer leaves fherma_h2d or fherma_d2h out: each says it
// is absent, so the loop neither clocks nor reports a transfer that did not
// happen. The answer's own, defined anywhere in it, replaces them when linked.
namespace fherma_device { bool h2d = true, d2h = true; }
__attribute__((weak)) void fherma_h2d(void*, const fherma::Inputs&) { fherma_device::h2d = false; }
__attribute__((weak)) fherma::Outputs fherma_d2h(void*, fherma::Outputs answer) {
    fherma_device::d2h = false;
    return answer;
}
// FHERMA_DEVICE_END

namespace {

// `,"name":seconds` since a mark, for the fields a case row may carry.
std::string clocked(const char* name, std::chrono::steady_clock::time_point mark) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9) << ",\"" << name << "\":"
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
    return out.str();
}

// A case row with more fields, put before its closing brace.
std::string with(std::string row, const std::string& more) {
    row.insert(row.size() - 1, more);
    return row;
}

std::string slurp(const fs::path& at) {
    std::ifstream file(at, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + at.string());
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

// A scanner for our own manifest and nothing else. The schema is two things —
// the point and a count — so this beats a dependency that would have to be
// fetched at build time inside a sandbox with no network.
double number(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const size_t at = text.find(quoted);
    if (at == std::string::npos) throw std::runtime_error("no " + key + " in manifest.json");
    return std::strtod(text.c_str() + text.find(':', at + quoted.size()) + 1, nullptr);
}

std::string numbered(size_t i) {
    std::ostringstream out;
    out << std::setw(6) << std::setfill('0') << i;
    return out.str();
}

template <class T>
fherma::Tensor<T> read(const fs::path& where, const std::string& name,
                       std::vector<int64_t> shape) {
    fherma::Tensor<T> tensor;
    tensor.shape = std::move(shape);

    const std::string raw = slurp(where / (name + ".bin"));
    const size_t count = static_cast<size_t>(tensor.count());
    if (raw.size() != count * sizeof(T)) {
        throw std::runtime_error(name + ": " + std::to_string(raw.size()) +
                                 " bytes for " + std::to_string(count) + " values");
    }

    // Little-endian on the wire, and on every platform this runs on, so the
    // copy is the decode.
    tensor.data.resize(count);
    std::memcpy(tensor.data.data(), raw.data(), raw.size());
    return tensor;
}

template <class T>
void write(const fs::path& where, const std::string& name,
           const fherma::Tensor<T>& tensor) {
    std::ofstream out(where / (name + ".bin"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(tensor.data.data()),
              static_cast<std::streamsize>(tensor.data.size() * sizeof(T)));
}

// A scalar on the wire is a tensor without dimensions: one value of its
// width, in its own file.
template <class T>
T read_one(const fs::path& where, const std::string& name) {
    const std::string raw = slurp(where / (name + ".bin"));
    if (raw.size() != sizeof(T)) {
        throw std::runtime_error(name + ": " + std::to_string(raw.size()) +
                                 " bytes for one value");
    }
    T value;
    std::memcpy(&value, raw.data(), sizeof(T));
    return value;
}

template <class T>
void write_one(const fs::path& where, const std::string& name, T value) {
    std::ofstream out(where / (name + ".bin"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Written after every case, not at the end: a process killed on its timeout
// has still done the cases before it, and a file written once at the end would
// throw them away.
void report(const fs::path& out, double init_s, const std::vector<std::string>& cases) {
    std::ofstream file(out / "results.json");
    file << std::fixed << std::setprecision(9);
    file << "{\"init_s\":" << init_s << ",\"cases\":[";
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i) file << ",";
        file << cases[i];
    }
    file << "]}";
}

std::string ok(size_t i, double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    out << "{\"i\":" << i << ",\"seconds\":" << seconds << ",\"status\":\"ok\"}";
    return out.str();
}

std::string crashed(size_t i, const std::string& why) {
    std::ostringstream out;
    out << "{\"i\":" << i << ",\"seconds\":null,\"status\":\"crashed\",\"note\":\"";
    for (char c : why.substr(0, 200)) {
        if (c == '"' || c == '\\') out << '\\';
        out << (c == '\n' ? ' ' : c);
    }
    out << "\"}";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: solution <point directory>" << std::endl;
        return 2;
    }

    const fs::path root = argv[1];
    const std::string manifest = slurp(root / "manifest.json");

    fherma::Point p{};
        p.B = static_cast<decltype(p.B)>(number(manifest, "B"));
        p.N = static_cast<decltype(p.N)>(number(manifest, "N"));
        p.W = static_cast<decltype(p.W)>(number(manifest, "W"));
        p.L = static_cast<decltype(p.L)>(number(manifest, "L"));
        p.q = read<uint32_t>(root / "point", "q", {static_cast<int64_t>(p.L)});
    const size_t total = static_cast<size_t>(number(manifest, "cases"));

    const fs::path answers_root = root / "out";
    fs::create_directories(answers_root);

    const auto setup = std::chrono::steady_clock::now();
    void* state = fherma_init(p);
    const std::chrono::duration<double> init_took =
        std::chrono::steady_clock::now() - setup;

    std::vector<std::string> cases;
    report(answers_root, init_took.count(), cases);

    for (size_t i = 0; i < total; ++i) {
        const fs::path where = root / "cases" / numbered(i);
        const fs::path answers = answers_root / numbered(i);

        fherma::Inputs in{};
        try {
            in.a = read<uint32_t>(where, "a", {static_cast<int64_t>(p.B), static_cast<int64_t>(p.N), static_cast<int64_t>(p.L)});
            in.b = read<uint32_t>(where, "b", {static_cast<int64_t>(p.B), static_cast<int64_t>(p.N), static_cast<int64_t>(p.L)});
        } catch (const std::exception& failure) {
            cases.push_back(crashed(i, std::string("reading the case: ") + failure.what()));
            report(answers_root, init_took.count(), cases);
            continue;
        }

        double seconds = 0;
        fherma::Outputs answer{};
        std::string moved;
        // FHERMA_H2D_BEGIN
        try {
            const auto mark = std::chrono::steady_clock::now();
            fherma_h2d(state, in);
            if (fherma_device::h2d) moved += clocked("h2d_s", mark);
        } catch (const std::exception& failure) {
            cases.push_back(crashed(i, std::string("fherma_h2d: ") + failure.what()));
            report(answers_root, init_took.count(), cases);
            continue;
        }
        // FHERMA_H2D_END
        try {
            // Monotonic, and around the call and nothing else.
            const auto started = std::chrono::steady_clock::now();
            answer = fherma_run(state, in);
            const std::chrono::duration<double> took =
                std::chrono::steady_clock::now() - started;
            seconds = took.count();
        } catch (const std::exception& failure) {
            cases.push_back(crashed(i, failure.what()));
            report(answers_root, init_took.count(), cases);
            continue;
        }
        // FHERMA_D2H_BEGIN
        try {
            const auto mark = std::chrono::steady_clock::now();
            answer = fherma_d2h(state, std::move(answer));
            if (fherma_device::d2h) moved += clocked("d2h_s", mark);
        } catch (const std::exception& failure) {
            cases.push_back(crashed(i, std::string("fherma_d2h: ") + failure.what()));
            report(answers_root, init_took.count(), cases);
            continue;
        }
        // FHERMA_D2H_END

        try {
            fs::create_directories(answers);
            write(answers, "c", answer.c);
        } catch (const std::exception& failure) {
            cases.push_back(crashed(i, std::string("writing the answer: ") + failure.what()));
            report(answers_root, init_took.count(), cases);
            continue;
        }

        cases.push_back(with(ok(i, seconds), moved));
        report(answers_root, init_took.count(), cases);
    }

    fherma_free(state);
    report(answers_root, init_took.count(), cases);
    return 0;
}
