// Agnostic in-tree port of upstream's
// mlir/lib/Transforms/Utils/CFGToSCF.cpp (Bahmann/Reissmann 2015,
// "Perfect Reconstructability of Control Flow from Demand Dependence
// Graphs"). Drives the lift entirely through mlir_api.h so the same
// implementation runs against both backends.
//
// Status: scaffolding only. The transformation entry currently returns
// false. See the TODO comments inside mlir_lift_cf_to_scf.c for the
// remaining work to land the three transforms (cycle lifting, branch
// lifting, return-like exit combining).

#ifndef MLIR_LIFT_CF_TO_SCF_H
#define MLIR_LIFT_CF_TO_SCF_H

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lift cf.* terminators in every func.func and llvm.func body of the
// `module` op into scf.* structured control flow. Mirrors the upstream
// MLIR_LiftCfToScf contract: returns true on success, false on failure
// or on input the algorithm cannot handle. On success there are no
// remaining cf.br / cf.cond_br / cf.switch ops other than those left
// behind for multi-kind return-like dispatches (which the algorithm
// proves cannot be lifted further).
bool MLIR_LiftCfToScfNative(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif

#endif // MLIR_LIFT_CF_TO_SCF_H
