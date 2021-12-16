// Package parse contains the generic code generation capabilities
// that power genny.
//
//     genny gen "{types}"
//
//     gen - generates type specific code (to stdout) from generic code (via stdin)
//
//     {types}  - (required) Specific types for each generic type in the source
//     {types} format:  {generic}={specific}[,another][ {generic2}={specific2}]
//     Examples:
//       Generic=Specific
//       Generic1=Specific1 Generic2=Specific2
//       Generic1=Specific1,Specific2 Generic2=Specific3,Specific4
package parse
