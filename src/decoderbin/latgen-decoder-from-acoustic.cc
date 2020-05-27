#include <iostream>
#include <decoder/decodable-matrix.h>
#include <fstext/fstext-lib.h>
#include <fstext/kaldi-fst-io-inl.h>
#include "decoder/lattice-faster-decoder.h"
#include "util/common-utils.h"
#include "ThreadPool.h"

using namespace kaldi;

int Usage(const char* program) {
    std::cerr << "Usage: " << program << "acoustic_scale acoustic graph" << std::endl;
    return EXIT_FAILURE;
}

fst::VectorFst<LatticeArc> Decode(const fst::StdVectorFst &graph, DecodableInterface *decodable) {
    LatticeFasterDecoderConfig config;
    config.max_active = 2000;
    LatticeFasterDecoder decoder{graph, config};

    if (!decoder.Decode(decodable))
        KALDI_ERR << "Failed to decode";
    fst::VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded))
        KALDI_ERR << "Failed to get best path from the decoder";

    delete decodable;
    return decoded;
}

int main(int argc, const char** argv) {
    if (argc != 4) return Usage(argv[0]);
    const auto acoustic_scale = std::stof(argv[1]);
    const std::string feat_rspecifier{argv[2]};
    SequentialGeneralMatrixReader reader{feat_rspecifier};
    const auto graph = fst::StdVectorFst::Read(argv[3]);

    const auto processor_count = std::thread::hardware_concurrency();
    ThreadPool pool{processor_count};
    std::vector<std::future<fst::VectorFst<LatticeArc>>> results;
    std::vector<Matrix<BaseFloat>*> matrices;
    for (; !reader.Done(); reader.Next()) {
        auto llk = reader.Value().GetFullMatrix();
        auto matrix = new Matrix<BaseFloat>{llk.NumRows(), llk.NumCols()};
        matrix->CopyFromMat(llk);
        matrices.push_back(matrix);
        auto decodable = new DecodableMatrixScaled{*matrix, acoustic_scale};
        if (decodable->NumIndices() == 0 || decodable->NumFramesReady() == 0)
            KALDI_ERR << "Invalid decodable";
        results.push_back(pool.enqueue(Decode, *graph, decodable));
    }

    for (int i = 0; i < results.size(); ++i) {
        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        GetLinearSymbolSequence(results.at(i).get(), &alignment, &words, &weight);
        delete matrices.at(i);

        for (int & word : words) {
            std::string s = graph->OutputSymbols()->Find(word);
            if (s.empty())
                KALDI_ERR << "Word-id " << word << " not in symbol table.";
            std::cout << s << ' ';
        }
        std::cout << '\n';
    }
    return 0;
}