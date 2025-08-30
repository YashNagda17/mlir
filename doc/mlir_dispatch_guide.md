# MLIR Operation Dispatching Methods Guide

This document covers different approaches for dispatching operations in MLIR, with code examples and trade-offs for each method.

## Overview

When processing MLIR operations, you often need to perform different actions based on the operation type. MLIR provides several mechanisms for this dispatching, each with different trade-offs in terms of performance, flexibility, and maintainability.

---

## 1. TypeSwitch (LLVM ADT)

The most idiomatic MLIR approach using LLVM's `TypeSwitch` utility for clean pattern matching.

### Code Example

```cpp
#include "llvm/ADT/TypeSwitch.h"

void dispatchUsingTypeSwitch(Operation *op) {
  TypeSwitch<Operation*>(op)
    .Case<arith::AddIOp>([](arith::AddIOp addOp) {
      printf("  Found AddI operation\n");
    })
    .Case<arith::MulIOp>([](arith::MulIOp mulOp) {
      printf("  Found MulI operation\n");
    })
    .Case<arith::ConstantOp>([](arith::ConstantOp constOp) {
      printf("  Found Constant operation\n");
    })
    .Case<func::FuncOp, func::ReturnOp>([](Operation *op) {
      // Can handle multiple types in one case
      printf("  Found Function-related operation\n");
    })
    .Default([](Operation *op) {
      printf("  Unknown operation: %s\n", 
             op->getName().getStringRef().str().c_str());
    });
}
```

### Pros
- Type-safe with compile-time checking
- Clean, readable syntax
- Can return values from the switch
- Supports multiple types in one case
- Follows MLIR/LLVM coding conventions

### Cons
- Compile-time fixed (cannot add cases dynamically)
- All cases must be known at compile time
- Requires including specific operation headers

---

## 2. dyn_cast Chain (Traditional If-Else)

The traditional approach using LLVM's casting infrastructure.

### Code Example

```cpp
void dispatchUsingDynCast(Operation *op) {
  if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
    printf("  %s = arith.addi ", getValueName(addOp.getResult()).c_str());
    printf("%s, ", getValueName(addOp.getLhs()).c_str());
    printf("%s : ", getValueName(addOp.getRhs()).c_str());
    printType(addOp.getType());
    printf("\n");
  } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    printf("  %s = arith.muli ", getValueName(mulOp.getResult()).c_str());
    printf("%s, ", getValueName(mulOp.getLhs()).c_str());
    printf("%s : ", getValueName(mulOp.getRhs()).c_str());
    printType(mulOp.getType());
    printf("\n");
  } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    printf("  %s = arith.constant ", getValueName(constantOp.getResult()).c_str());
    // ... handle constant
  } else {
    printf("  <unknown operation: %s>\n", 
           op->getName().getStringRef().str().c_str());
  }
}
```

### Pros
- Simple and straightforward
- Direct access to typed operations
- Early exit possible with returns
- Familiar C++ pattern

### Cons
- Can become verbose with many operation types
- Less elegant than TypeSwitch
- Performance degrades linearly with number of types

---

## 3. String-based Operation Names

Runtime dispatch using operation name strings.

### Code Example

```cpp
void dispatchUsingOperationName(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  
  // Using a map of operation names to handlers
  static std::unordered_map<std::string, 
                            std::function<void(Operation*)>> handlers = {
    {"arith.addi", [](Operation *op) { 
      printf("  Processing integer addition\n");
    }},
    {"arith.muli", [](Operation *op) { 
      printf("  Processing integer multiplication\n");
    }},
    {"arith.constant", [](Operation *op) { 
      printf("  Processing constant\n");
    }},
    {"func.func", [](Operation *op) { 
      printf("  Processing function\n");
    }},
    {"func.return", [](Operation *op) { 
      printf("  Processing return\n");
    }}
  };
  
  auto it = handlers.find(opName.str());
  if (it != handlers.end()) {
    it->second(op);
  } else {
    printf("  No handler for: %s\n", opName.str().c_str());
  }
}
```

### Pros
- Fully dynamic (can add handlers at runtime)
- Can handle unknown operations gracefully
- Easy to extend without recompilation
- Good for plugin systems

### Cons
- No compile-time type checking
- String comparison overhead
- Loss of type safety (requires manual casting)
- Potential for typos in operation names

---

## 4. TypeID Comparison

Fast type comparison using MLIR's TypeID system.

### Code Example

```cpp
void dispatchUsingTypeID(Operation *op) {
  TypeID opTypeID = op->getName().getTypeID();
  
  // Build a static map of TypeIDs to handlers
  static std::unordered_map<TypeID, 
                            std::function<void(Operation*)>> typeIDHandlers = {
    {TypeID::get<arith::AddIOp>(), [](Operation *op) {
      auto addOp = cast<arith::AddIOp>(op);
      printf("  AddI with TypeID matching\n");
    }},
    {TypeID::get<arith::MulIOp>(), [](Operation *op) {
      auto mulOp = cast<arith::MulIOp>(op);
      printf("  MulI with TypeID matching\n");
    }},
    {TypeID::get<arith::ConstantOp>(), [](Operation *op) {
      auto constOp = cast<arith::ConstantOp>(op);
      printf("  Constant with TypeID matching\n");
    }}
  };
  
  auto it = typeIDHandlers.find(opTypeID);
  if (it != typeIDHandlers.end()) {
    it->second(op);
  } else {
    printf("  No TypeID handler found\n");
  }
}
```

### Pros
- Faster than string comparison
- Unique identifiers for each operation type
- Can be used with hash maps for O(1) lookup
- Type-safe at registration time

### Cons
- Still requires a mapping structure
- Less readable than TypeSwitch
- Requires manual casting inside handlers

---

## 5. Visitor Pattern

Encapsulated traversal and dispatch logic using the visitor pattern.

### Code Example

```cpp
class OperationPrinterVisitor {
public:
  void visit(Operation *op) {
    op->walk([this](Operation *nestedOp) {
      return this->visitOperation(nestedOp);
    });
  }
  
private:
  WalkResult visitOperation(Operation *op) {
    TypeSwitch<Operation*, WalkResult>(op)
      .Case<arith::AddIOp>([this](arith::AddIOp op) { 
        return visitAddI(op); 
      })
      .Case<arith::MulIOp>([this](arith::MulIOp op) { 
        return visitMulI(op); 
      })
      .Default([](Operation *op) {
        printf("  Visiting: %s\n", 
               op->getName().getStringRef().str().c_str());
        return WalkResult::advance();
      });
    
    return WalkResult::advance();
  }
  
  WalkResult visitAddI(arith::AddIOp op) {
    printf("  Visitor: Found AddI\n");
    return WalkResult::advance();
  }
  
  WalkResult visitMulI(arith::MulIOp op) {
    printf("  Visitor: Found MulI\n");
    return WalkResult::advance();
  }
};
```

### Pros
- Clean separation of concerns
- Good for tree traversal (with `walk()`)
- Encapsulates state and behavior
- Extensible through inheritance

### Cons
- More boilerplate code
- Still needs internal dispatch mechanism
- Can be overkill for simple cases
- Indirection can impact performance

---

## 6. Interface-based Dispatch

Leverages MLIR's operation interfaces and traits system.

### Code Example

```cpp
void dispatchUsingInterfaces(Operation *op) {
  // Check if operation implements certain interfaces
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    printf("  Operation has memory effects interface\n");
  }
  
  if (auto sideEffect = dyn_cast<ConditionallySpeculatable>(op)) {
    printf("  Operation implements ConditionallySpeculatable\n");
  }
  
  // Arithmetic operations implement ArithFastMathInterface
  if (auto mathOp = dyn_cast<arith::ArithFastMathInterface>(op)) {
    printf("  Operation implements ArithFastMathInterface\n");
  }
  
  // You can also check for traits
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    printf("  Operation is a terminator\n");
  }
  
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    printf("  Operation is constant-like\n");
  }
}
```

### Pros
- Polymorphic behavior across operation types
- Groups related operations logically
- Follows OOP principles
- Good for cross-cutting concerns

### Cons
- Only works for operations implementing specific interfaces
- May require multiple interface checks
- Not suitable for operation-specific logic
- Interface implementation overhead

---

## 7. Dialect-based Dispatch

First-level dispatch by dialect, then operation-specific handling.

### Code Example

```cpp
void dispatchUsingDialect(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect) {
    printf("  Operation has no registered dialect\n");
    return;
  }
  
  StringRef dialectName = dialect->getNamespace();
  
  if (dialectName == "arith") {
    printf("  Arithmetic dialect operation: %s\n", 
           op->getName().getStringRef().str().c_str());
    
    // Further dispatch within arithmetic dialect
    if (isa<arith::AddIOp>(op)) {
      printf("    Specifically an integer add\n");
    } else if (isa<arith::AddFOp>(op)) {
      printf("    Specifically a float add\n");
    }
  } else if (dialectName == "func") {
    printf("  Function dialect operation: %s\n",
           op->getName().getStringRef().str().c_str());
  } else {
    printf("  Other dialect: %s\n", dialectName.str().c_str());
  }
}
```

### Pros
- Good for dialect-specific logic
- Hierarchical organization
- Can handle all operations from a dialect uniformly
- Useful for dialect conversion passes

### Cons
- Coarse-grained (still needs secondary dispatch)
- String comparison for dialect names
- Not efficient for mixed-dialect processing
- Requires knowledge of dialect structure

---

## 8. Attribute-based Dispatch

Uses operation attributes to determine behavior.

### Code Example

```cpp
void dispatchUsingAttributes(Operation *op) {
  // Dispatch based on presence of specific attributes
  if (op->hasAttr("custom.print_style")) {
    StringAttr styleAttr = op->getAttrOfType<StringAttr>("custom.print_style");
    if (styleAttr) {
      StringRef style = styleAttr.getValue();
      if (style == "verbose") {
        printf("  Verbose printing for: %s\n", 
               op->getName().getStringRef().str().c_str());
      } else if (style == "compact") {
        printf("  Compact printing for: %s\n",
               op->getName().getStringRef().str().c_str());
      }
    }
  }
  
  // Check for specific attribute patterns
  if (op->hasAttr("fastmath")) {
    printf("  Operation has fastmath flags\n");
  }
}
```

### Pros
- Very flexible and data-driven
- Can change behavior without code changes
- Good for configuration and customization
- Works across operation types

### Cons
- Requires attributes to be set appropriately
- Runtime overhead of attribute lookup
- No compile-time checking
- Can lead to scattered logic

---

## 9. Registration-based (Registry Pattern)

Dynamic registration of handlers for extensibility.

### Code Example

```cpp
class OperationPrinterRegistry {
private:
  using PrinterFunc = std::function<void(Operation*)>;
  std::unordered_map<TypeID, PrinterFunc> printers;
  
public:
  // Register a printer for a specific operation type
  template<typename OpTy>
  void registerPrinter(std::function<void(OpTy)> printer) {
    printers[TypeID::get<OpTy>()] = [printer](Operation *op) {
      printer(cast<OpTy>(op));
    };
  }
  
  // Dispatch to registered printer
  void print(Operation *op) {
    auto it = printers.find(op->getName().getTypeID());
    if (it != printers.end()) {
      it->second(op);
    } else {
      printf("  No registered printer for: %s\n",
             op->getName().getStringRef().str().c_str());
    }
  }
  
  // Allow dynamic registration
  void registerDynamicPrinter(StringRef opName, PrinterFunc printer) {
    // This would need the actual TypeID, simplified here
    printf("  Registered dynamic printer for: %s\n", opName.str().c_str());
  }
};

// Usage:
OperationPrinterRegistry registry;
registry.registerPrinter<arith::AddIOp>([](arith::AddIOp op) {
  printf("  Custom registered printer for AddI\n");
});
registry.print(someOperation);
```

### Pros
- Highly extensible
- Supports plugin architectures
- Separation of concerns
- Can mix compile-time and runtime registration
- Good for framework design

### Cons
- More complex setup required
- Indirection overhead
- Potential for registration order issues
- Memory overhead for registry storage

---

## Performance Comparison

| Method | Performance | Use Case |
|--------|------------|----------|
| TypeID comparison | Fastest | High-performance passes |
| dyn_cast/isa chains | Fast | Simple dispatching |
| TypeSwitch | Fast | General purpose |
| Interface checks | Moderate | Cross-cutting concerns |
| String comparison | Slow | Dynamic/plugin systems |
| Attribute lookup | Slowest | Configuration-driven |

---

## Recommendations

### For Production MLIR Passes
- **Primary choice**: TypeSwitch for clarity and performance
- **Alternative**: dyn_cast chains for simple cases with early returns

### For Extensible Systems
- **Primary choice**: Registration-based pattern
- **Alternative**: String-based dispatch with careful optimization

### For Analysis Passes
- **Primary choice**: Interface-based for semantic properties
- **Alternative**: Visitor pattern for complex traversals

### For Debug/Development Tools
- **Primary choice**: String-based for flexibility
- **Alternative**: Attribute-based for configurability

---

## Common Patterns in MLIR Codebase

1. **Small fixed sets of operations**: TypeSwitch is preferred
2. **Transformation passes**: Often use dyn_cast chains with early returns
3. **Generic analyses**: Interface-based or trait-based dispatch
4. **Dialect conversions**: Hierarchical dialect-then-operation dispatch
5. **Debug utilities**: String-based for handling unknown operations

---

## Example: Combining Multiple Approaches

```cpp
// Hybrid approach for maximum flexibility
void hybridDispatch(Operation *op) {
  // Fast path for known critical operations
  if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
    handleAddIOp(addOp);
    return;
  }
  
  // Interface-based for general properties
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    handleTerminator(op);
    return;
  }
  
  // TypeSwitch for remaining known operations
  TypeSwitch<Operation*>(op)
    .Case<arith::MulIOp, arith::DivIOp>([](Operation *op) {
      handleArithmeticOp(op);
    })
    .Default([](Operation *op) {
      // Fall back to string-based for unknown ops
      handleUnknownOp(op->getName().getStringRef());
    });
}
```

This hybrid approach optimizes for common cases while maintaining flexibility for unknown operations.