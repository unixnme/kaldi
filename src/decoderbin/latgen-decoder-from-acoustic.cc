#include <iostream>
#include <decoder/decodable-matrix.h>
#include <fstext/fstext-lib.h>
#include <fstext/kaldi-fst-io-inl.h>
#include "decoder/lattice-faster-decoder.h"
#include "util/common-utils.h"

int Usage(const char* program) {
    std::cerr << "Usage: " << program << "acoustic graph lattice" << std::endl;
    return EXIT_FAILURE;
}

int main(int argc, const char** argv) {
    using namespace kaldi;

    if (argc != 4) return Usage(argv[0]);
    const std::string feat_rspecifier{argv[1]};
    SequentialGeneralMatrixReader reader{feat_rspecifier};
    auto graph = fst::StdVectorFst::Read(argv[2]);

    LatticeFasterDecoderConfig config;
    LatticeFasterDecoder decoder{*graph, config};
    for (; !reader.Done(); reader.Next()) {
        DecodableMatrixScaled decodable{reader.Value().GetFullMatrix(), 1};
        if (!decoder.Decode(&decodable))
            KALDI_ERR << "Failed to decode";
        fst::VectorFst<LatticeArc> decoded;
        if (!decoder.GetBestPath(&decoded))
            KALDI_ERR << "Failed to get best path from the decoder";

        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        for (int & word : words) {
            std::string s = graph->OutputSymbols()->Find(word);
            if (s.empty())
                KALDI_ERR << "Word-id " << word << " not in symbol table.";
            std::cout << s << ' ';
        }
        std::cout << '\n';

//        for (auto frame = 0; !decodable.IsLastFrame(frame); ++frame) {
//            for (auto idx = 0; idx < decodable.NumIndices(); ++idx) {
//                std::cout << decodable.LogLikelihood(frame, idx + 1) << "\t";
//            }
//            std::cout << std::endl;
//        }
    }
    return 0;
}