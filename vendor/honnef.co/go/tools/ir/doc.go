// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ir defines a representation of the elements of Go programs
// (packages, types, functions, variables and constants) using a
// static single-information (SSI) form intermediate representation
// (IR) for the bodies of functions.
//
// THIS INTERFACE IS EXPERIMENTAL AND IS LIKELY TO CHANGE.
//
// For an introduction to SSA form, upon which SSI builds, see
// http://en.wikipedia.org/wiki/Static_single_assignment_form.
// This page provides a broader reading list:
// http://www.dcs.gla.ac.uk/~jsinger/ssa.html.
//
// For an introduction to SSI form, see The static single information
// form by C. Scott Ananian.
//
// The level of abstraction of the IR form is intentionally close to
// the source language to facilitate construction of source analysis
// tools.  It is not intended for machine code generation.
//
// The simplest way to create the IR of a package is
// to load typed syntax trees using golang.org/x/tools/go/packages, then
// invoke the irutil.Packages helper function. See ExampleLoadPackages
// and ExampleWholeProgram for examples.
// The resulting ir.Program contains all the packages and their
// members, but IR code is not created for function bodies until a
// subsequent call to (*Package).Build or (*Program).Build.
//
// The builder initially builds a naive IR form in which all local
// variables are addresses of stack locations with explicit loads and
// stores.  Registerisation of eligible locals and φ-node insertion
// using dominance and dataflow are then performed as a second pass
// called "lifting" to improve the accuracy and performance of
// subsequent analyses; this pass can be skipped by setting the
// NaiveForm builder flag.
//
// The primary interfaces of this package are:
//
//    - Member: a named member of a Go package.
//    - Value: an expression that yields a value.
//    - Instruction: a statement that consumes values and performs computation.
//    - Node: a Value or Instruction (emphasizing its membership in the IR value graph)
//
// A computation that yields a result implements both the Value and
// Instruction interfaces.  The following table shows for each
// concrete type which of these interfaces it implements.
//
//                      Value?          Instruction?    Member?
//   *Alloc             ✔               ✔
//   *BinOp             ✔               ✔
//   *BlankStore                        ✔
//   *Builtin           ✔
//   *Call              ✔               ✔
//   *ChangeInterface   ✔               ✔
//   *ChangeType        ✔               ✔
//   *Const             ✔               ✔
//   *Convert           ✔               ✔
//   *DebugRef                          ✔
//   *Defer             ✔               ✔
//   *Extract           ✔               ✔
//   *Field             ✔               ✔
//   *FieldAddr         ✔               ✔
//   *FreeVar           ✔
//   *Function          ✔                               ✔ (func)
//   *Global            ✔                               ✔ (var)
//   *Go                ✔               ✔
//   *If                                ✔
//   *Index             ✔               ✔
//   *IndexAddr         ✔               ✔
//   *Jump                              ✔
//   *Load              ✔               ✔
//   *MakeChan          ✔               ✔
//   *MakeClosure       ✔               ✔
//   *MakeInterface     ✔               ✔
//   *MakeMap           ✔               ✔
//   *MakeSlice         ✔               ✔
//   *MapLookup         ✔               ✔
//   *MapUpdate         ✔               ✔
//   *NamedConst                                        ✔ (const)
//   *Next              ✔               ✔
//   *Panic                             ✔
//   *Parameter         ✔               ✔
//   *Phi               ✔               ✔
//   *Range             ✔               ✔
//   *Recv              ✔               ✔
//   *Return                            ✔
//   *RunDefers                         ✔
//   *Select            ✔               ✔
//   *Send              ✔               ✔
//   *Sigma             ✔               ✔
//   *Slice             ✔               ✔
//   *Store             ✔               ✔
//   *StringLookup      ✔               ✔
//   *Type                                              ✔ (type)
//   *TypeAssert        ✔               ✔
//   *UnOp              ✔               ✔
//   *Unreachable                       ✔
//
// Other key types in this package include: Program, Package, Function
// and BasicBlock.
//
// The program representation constructed by this package is fully
// resolved internally, i.e. it does not rely on the names of Values,
// Packages, Functions, Types or BasicBlocks for the correct
// interpretation of the program.  Only the identities of objects and
// the topology of the IR and type graphs are semantically
// significant.  (There is one exception: Ids, used to identify field
// and method names, contain strings.)  Avoidance of name-based
// operations simplifies the implementation of subsequent passes and
// can make them very efficient.  Many objects are nonetheless named
// to aid in debugging, but it is not essential that the names be
// either accurate or unambiguous.  The public API exposes a number of
// name-based maps for client convenience.
//
// The ir/irutil package provides various utilities that depend only
// on the public API of this package.
//
// TODO(adonovan): Consider the exceptional control-flow implications
// of defer and recover().
//
// TODO(adonovan): write a how-to document for all the various cases
// of trying to determine corresponding elements across the four
// domains of source locations, ast.Nodes, types.Objects,
// ir.Values/Instructions.
//
package ir // import "honnef.co/go/tools/ir"
