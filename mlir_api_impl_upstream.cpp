// Upstream LLVM/MLIR-backed implementation of the public API in mlir_api.h.
//
// This is a minimal implementation sufficient for the cross-implementation
// smoke test in tests/cross/. It exercises:
//   * Module + unregistered op construction.
//   * The read-side surface used by mlir_generic_printer.c when there are
//     no operands, results, attributes, types, or values.
//
// Functions outside that minimal set are stubbed with UNIMPLEMENTED and
// will abort if called. Expanding coverage requires arena-allocated
// wrapper records for value types (mlir::Value/Type/Attribute) since
// those are value-semantics handles, not stable pointers.
//
// This file is compiled in "hosted" mode: corec headers are included
// without -DCOREC_FREESTANDING, so they fall through to system libc
// declarations. Linkage is against system libc/libc++ + libMLIR*.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"

extern "C" {
#include "mlir_api.h"
#include "mlir_op_names.h"
}

#define UNIMPLEMENTED()                                                   \
    do {                                                                  \
        std::fprintf(stderr, "%s unimplemented (upstream)\n", __func__);  \
        std::abort();                                                     \
    } while (0)

namespace {

// Global context owns all upstream MLIR objects we create.
struct UpstreamCtx {
    mlir::MLIRContext mctx;
    UpstreamCtx() {
        mctx.allowUnregisteredDialects(true);
    }
};

UpstreamCtx &globalCtx() {
    static UpstreamCtx g;
    return g;
}

// Convert between opaque integer handles and upstream pointers.
template <class T>
static inline uintptr_t toHandle(T *p) { return reinterpret_cast<uintptr_t>(p); }
template <class T>
static inline T *fromHandle(uintptr_t h) { return reinterpret_cast<T *>(h); }

// Map upstream op name → our enum (only entries the cross-test exercises).
static MLIR_OpType opTypeFromName(llvm::StringRef name) {
    if (name == "builtin.module") return OP_TYPE_MODULE;
    return OP_TYPE_UNREGISTERED;
}

} // namespace

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

extern "C" void MLIR_InitApi(MLIR_Context *ctx, MLIR_OpHandle root) {
    (void)ctx;
    (void)root;
}

extern "C" void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena) {
    ctx->arena = arena;
}

extern "C" Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx) {
    return ctx->arena;
}

// -----------------------------------------------------------------------------
// Region / Block construction
// -----------------------------------------------------------------------------

extern "C" MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *) {
    return toHandle(new mlir::Region());
}

extern "C" MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *) {
    return toHandle(new mlir::Block());
}

extern "C" void MLIR_AppendRegionBlock(MLIR_Context *, MLIR_RegionHandle r,
                                        MLIR_BlockHandle b) {
    auto *region = fromHandle<mlir::Region>(r);
    auto *block = fromHandle<mlir::Block>(b);
    region->push_back(block);
}

extern "C" void MLIR_AppendBlockOp(MLIR_Context *, MLIR_BlockHandle b,
                                    MLIR_OpHandle op) {
    auto *block = fromHandle<mlir::Block>(b);
    auto *operation = fromHandle<mlir::Operation>(op);
    block->push_back(operation);
}

extern "C" void MLIR_AppendBlockArg(MLIR_Context *, MLIR_BlockHandle,
                                     MLIR_ValueHandle) {
    UNIMPLEMENTED();
}

// -----------------------------------------------------------------------------
// Op construction
// -----------------------------------------------------------------------------

extern "C" MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *, MLIR_OpType type, string opname,
    MLIR_AttributeHandle * /*attrs*/, size_t n_attrs,
    MLIR_TypeHandle * /*result_types*/, size_t n_result_types,
    MLIR_ValueHandle * /*results*/, size_t n_results,
    MLIR_ValueHandle * /*operands*/, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle, MLIR_LocationHandle, string, int64_t) {
    auto &ctx = globalCtx().mctx;

    if (n_attrs != 0 || n_result_types != 0 || n_results != 0 ||
        n_operands != 0) {
        std::fprintf(stderr,
                     "MLIR_CreateOp: upstream impl only supports the minimal "
                     "no-operands/results/attrs path\n");
        std::abort();
    }

    // Resolve op name. For OP_TYPE_MODULE the C side uses "module" but
    // upstream's canonical name is "builtin.module".
    std::string name(opname.str, opname.size);
    if (type == OP_TYPE_MODULE && name == "module") {
        name = "builtin.module";
    }

    mlir::OperationState state(mlir::UnknownLoc::get(&ctx),
                               llvm::StringRef(name));
    for (size_t i = 0; i < n_regions; i++) {
        auto *region = fromHandle<mlir::Region>(regions[i]);
        state.addRegion(std::unique_ptr<mlir::Region>(region));
    }
    mlir::Operation *op = mlir::Operation::create(state);
    return toHandle(op);
}

extern "C" void MLIR_AppendOpAttribute(MLIR_Context *, MLIR_OpHandle,
                                        MLIR_AttributeHandle) {
    UNIMPLEMENTED();
}

// -----------------------------------------------------------------------------
// Op accessors
// -----------------------------------------------------------------------------

extern "C" MLIR_OpType MLIR_GetOpType(MLIR_OpHandle h) {
    auto *op = fromHandle<mlir::Operation>(h);
    return opTypeFromName(op->getName().getStringRef());
}

extern "C" string MLIR_GetOpName(MLIR_OpHandle h) {
    auto *op = fromHandle<mlir::Operation>(h);
    auto sr = op->getName().getStringRef();
    string s;
    s.str = const_cast<char *>(sr.data());
    s.size = sr.size();
    return s;
}

extern "C" string MLIR_GetOpName_string(MLIR_OpHandle h) {
    auto *op = fromHandle<mlir::Operation>(h);
    auto sr = op->getName().getStringRef();
    // For OP_TYPE_MODULE the C side prints "module"; strip "builtin." prefix.
    if (sr == "builtin.module") {
        string s;
        s.str = const_cast<char *>("module");
        s.size = 6;
        return s;
    }
    string s;
    s.str = const_cast<char *>(sr.data());
    s.size = sr.size();
    return s;
}

extern "C" MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle) {
    return MLIR_INVALID_HANDLE;
}
extern "C" string MLIR_GetOpTrailingComment(MLIR_OpHandle) {
    string s; s.str = nullptr; s.size = 0; return s;
}
extern "C" int64_t MLIR_GetOpSourceLineStart(MLIR_OpHandle) { return -1; }
extern "C" MLIR_LocationHandle MLIR_GetOpUnnumberedLocationDef(MLIR_OpHandle) {
    return MLIR_INVALID_HANDLE;
}

extern "C" size_t MLIR_GetOpNumOperands(MLIR_OpHandle h) {
    return fromHandle<mlir::Operation>(h)->getNumOperands();
}
extern "C" MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle, size_t) {
    UNIMPLEMENTED();
}
extern "C" size_t MLIR_GetOpNumResults(MLIR_OpHandle h) {
    return fromHandle<mlir::Operation>(h)->getNumResults();
}
extern "C" MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle, size_t) {
    UNIMPLEMENTED();
}
extern "C" size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle h) {
    return fromHandle<mlir::Operation>(h)->getNumResults();
}
extern "C" MLIR_TypeHandle MLIR_GetOpResult_type(MLIR_OpHandle, size_t) {
    UNIMPLEMENTED();
}
extern "C" size_t MLIR_GetOpNumAttributes(MLIR_OpHandle h) {
    return fromHandle<mlir::Operation>(h)->getAttrs().size();
}
extern "C" MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle, size_t) {
    UNIMPLEMENTED();
}
extern "C" size_t MLIR_GetOpNumRegions(MLIR_OpHandle h) {
    return fromHandle<mlir::Operation>(h)->getNumRegions();
}
extern "C" MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle h, size_t i) {
    return toHandle(&fromHandle<mlir::Operation>(h)->getRegion(i));
}

// -----------------------------------------------------------------------------
// Region accessors
// -----------------------------------------------------------------------------

extern "C" size_t MLIR_GetRegionNumBlocks(MLIR_RegionHandle h) {
    auto *region = fromHandle<mlir::Region>(h);
    size_t n = 0;
    for (auto it = region->begin(); it != region->end(); ++it) ++n;
    return n;
}
extern "C" MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_RegionHandle h, size_t i) {
    auto *region = fromHandle<mlir::Region>(h);
    auto it = region->begin();
    for (size_t k = 0; k < i; k++) ++it;
    return toHandle(&*it);
}

// -----------------------------------------------------------------------------
// Block accessors
// -----------------------------------------------------------------------------

extern "C" size_t MLIR_GetBlockNumOps(MLIR_BlockHandle h) {
    auto *block = fromHandle<mlir::Block>(h);
    size_t n = 0;
    for (auto &op : *block) { (void)op; ++n; }
    return n;
}
extern "C" MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle h, size_t i) {
    auto *block = fromHandle<mlir::Block>(h);
    auto it = block->begin();
    for (size_t k = 0; k < i; k++) ++it;
    return toHandle(&*it);
}
extern "C" size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle h) {
    return fromHandle<mlir::Block>(h)->getNumArguments();
}
extern "C" MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle, size_t) {
    UNIMPLEMENTED();
}

// -----------------------------------------------------------------------------
// Stubs — value/type/attribute surfaces are not exercised by the minimal driver
// -----------------------------------------------------------------------------

extern "C" MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *, string,
                                                     uint32_t, MLIR_TypeHandle,
                                                     MLIR_LocationHandle) {
    UNIMPLEMENTED();
}
extern "C" MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *, MLIR_OpHandle,
                                                     uint32_t, MLIR_TypeHandle,
                                                     string, MLIR_LocationHandle) {
    UNIMPLEMENTED();
}
extern "C" MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle) { UNIMPLEMENTED(); }
extern "C" string MLIR_GetValueRegisterName(MLIR_ValueHandle) { UNIMPLEMENTED(); }
extern "C" uint32_t MLIR_GetValueResultIndex(MLIR_ValueHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_ValueHandle) { UNIMPLEMENTED(); }

extern "C" string MLIR_GetTypeString(MLIR_Context *, MLIR_TypeHandle) { UNIMPLEMENTED(); }

// MLIR_OpTypeToString is provided by mlir_op_names.c and shared with the
// native impl, so we don't redefine it here. (Linked via mlir_op_names.c.)
extern "C" string MLIR_MLIR_OpTypeToString(MLIR_OpType type) {
    return op_type_to_string(type);
}

extern "C" size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle, size_t) { UNIMPLEMENTED(); }
extern "C" size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle, size_t) { UNIMPLEMENTED(); }
extern "C" int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle) { UNIMPLEMENTED(); }
extern "C" MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle) { UNIMPLEMENTED(); }
extern "C" string MLIR_GetAttributeName(MLIR_AttributeHandle) { UNIMPLEMENTED(); }
extern "C" string MLIR_GetAttributeString(MLIR_AttributeHandle) { UNIMPLEMENTED(); }

// Location-map helpers — printer doesn't use them in the minimal path.
extern "C" size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *) { return 0; }
extern "C" size_t MLIR_CollectLocationMap(const MLIR_LocationMap *, string *,
                                           MLIR_LocationHandle *, size_t) {
    return 0;
}

// Hosted entry point: defer to app_main() defined in driver.c.
extern "C" int app_main(void);
extern "C" void platform_init(int argc, char **argv);
int main(int /*argc*/, char ** /*argv*/) {
    platform_init(0, nullptr);
    return app_main();
}
