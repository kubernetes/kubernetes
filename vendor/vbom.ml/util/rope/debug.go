package rope

// debug enables debug output.
const debug = true

// MarkGoStringedRope is a flag that, when enabled, prepends "/*Rope*/ " to the
// result of Rope.GoString(). This can be useful when debugging code using
// interface{} values, where Ropes and strings can coexist.
//
// This should only ever be changed from init() functions, to ensure there are
// no data races.
var MarkGoStringedRope = true
