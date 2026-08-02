// Shared integer/float literal helpers used by both mlir_api_impl.c (native
// backend) and mlir_api_impl_upstream.cpp (upstream backend).

#include <stdint.h>
#include <string.h>

#include "mlir_api.h"

static uint32_t mlir_literal_words_for_width(uint32_t width) {
    if (width == 0) return 0;
    return (width + 63u) / 64u;
}

static void mlir_literal_clear_larger_words(uint64_t *dst) {
    for (int i = 0; i < MLIR_LITERAL_LARGER_BITS_WORDS; i++) dst[i] = 0;
}

static int64_t mlir_literal_sign_extend_from_word(uint32_t width,
                                                  uint64_t word0) {
    if (width >= 64) return (int64_t)word0;
    uint64_t mask = (1ULL << width) - 1ULL;
    uint64_t v = word0 & mask;
    if (width > 0 && (v >> (width - 1)) & 1ULL)
        v |= ~mask;
    return (int64_t)v;
}

bool MLIR_IntegerLiteral_uses_larger_bits(const MLIR_IntegerLiteral *lit) {
    return lit && lit->is_larger_bits;
}

bool MLIR_IntegerLiteral_set_value(MLIR_IntegerLiteral *lit, uint32_t width,
                                   int64_t value) {
    if (!lit || width == 0 || width > 64) return false;
    lit->kind = MLIR_LITERAL_INTEGER;
    lit->width = width;
    lit->value = value;
    lit->is_larger_bits = false;
    lit->larger_word_count = 0;
    mlir_literal_clear_larger_words(lit->value_larger_bits);
    return true;
}

bool MLIR_IntegerLiteral_set_larger_bits(MLIR_IntegerLiteral *lit,
                                         uint32_t width, const uint64_t *words,
                                         uint32_t word_count) {
    if (!lit || !words || width == 0) return false;
    uint32_t need = mlir_literal_words_for_width(width);
    if (need == 0 || need > MLIR_LITERAL_LARGER_BITS_WORDS) return false;
    if (word_count < need) return false;
    lit->kind = MLIR_LITERAL_INTEGER;
    lit->width = width;
    lit->value = 0;
    lit->is_larger_bits = true;
    lit->larger_word_count = need;
    mlir_literal_clear_larger_words(lit->value_larger_bits);
    for (uint32_t i = 0; i < need; i++) lit->value_larger_bits[i] = words[i];
    return true;
}

bool MLIR_IntegerLiteral_get_value(const MLIR_IntegerLiteral *lit,
                                   int64_t *out) {
    if (!lit || !out) return false;
    if (!lit->is_larger_bits) {
        *out = lit->value;
        return true;
    }
    if (lit->width > 64 || lit->larger_word_count == 0) return false;
    *out = mlir_literal_sign_extend_from_word(lit->width,
                                              lit->value_larger_bits[0]);
    return true;
}

bool MLIR_IntegerLiteral_get_larger_bits(const MLIR_IntegerLiteral *lit,
                                         uint64_t *words, uint32_t words_cap,
                                         uint32_t *out_word_count) {
    if (!lit || !out_word_count) return false;
    if (!lit->is_larger_bits) return false;
    if (words) {
        if (words_cap < lit->larger_word_count) return false;
        for (uint32_t i = 0; i < lit->larger_word_count; i++)
            words[i] = lit->value_larger_bits[i];
    }
    *out_word_count = lit->larger_word_count;
    return true;
}

bool MLIR_FloatLiteral_uses_larger_bits(const MLIR_FloatLiteral *lit) {
    return lit && lit->is_larger_bits;
}

bool MLIR_FloatLiteral_set_value(MLIR_FloatLiteral *lit, uint32_t width,
                                 MLIR_FloatEncoding encoding,
                                 double value) {
    if (!lit || width == 0 || width > 64) return false;
    lit->kind = MLIR_LITERAL_FLOAT;
    lit->width = width;
    lit->encoding = encoding;
    lit->value = value;
    lit->is_larger_bits = false;
    lit->larger_word_count = 0;
    mlir_literal_clear_larger_words(lit->value_larger_bits);
    return true;
}

bool MLIR_FloatLiteral_set_larger_bits(MLIR_FloatLiteral *lit, uint32_t width,
                                       MLIR_FloatEncoding encoding,
                                       const uint64_t *words,
                                       uint32_t word_count) {
    if (!lit || !words || width == 0) return false;
    uint32_t need = mlir_literal_words_for_width(width);
    if (need == 0 || need > MLIR_LITERAL_LARGER_BITS_WORDS) return false;
    if (word_count < need) return false;
    lit->kind = MLIR_LITERAL_FLOAT;
    lit->width = width;
    lit->encoding = encoding;
    lit->value = 0.0;
    lit->is_larger_bits = true;
    lit->larger_word_count = need;
    mlir_literal_clear_larger_words(lit->value_larger_bits);
    for (uint32_t i = 0; i < need; i++) lit->value_larger_bits[i] = words[i];
    return true;
}

bool MLIR_FloatLiteral_get_value(const MLIR_FloatLiteral *lit, double *out) {
    if (!lit || !out) return false;
    if (lit->is_larger_bits) return false;
    *out = lit->value;
    return true;
}

bool MLIR_FloatLiteral_get_larger_bits(const MLIR_FloatLiteral *lit,
                                       uint64_t *words, uint32_t words_cap,
                                       uint32_t *out_word_count) {
    if (!lit || !out_word_count) return false;
    if (!lit->is_larger_bits) return false;
    if (words) {
        if (words_cap < lit->larger_word_count) return false;
        for (uint32_t i = 0; i < lit->larger_word_count; i++)
            words[i] = lit->value_larger_bits[i];
    }
    *out_word_count = lit->larger_word_count;
    return true;
}
