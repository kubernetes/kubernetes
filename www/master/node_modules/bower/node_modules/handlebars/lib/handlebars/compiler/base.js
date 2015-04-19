import parser from "./parser";
import AST from "./ast";
module Helpers from "./helpers";
import { extend } from "../utils";

export { parser };

var yy = {};
extend(yy, Helpers, AST);

export function parse(input) {
  // Just return if an already-compile AST was passed in.
  if (input.constructor === AST.ProgramNode) { return input; }

  parser.yy = yy;

  return parser.parse(input);
}
