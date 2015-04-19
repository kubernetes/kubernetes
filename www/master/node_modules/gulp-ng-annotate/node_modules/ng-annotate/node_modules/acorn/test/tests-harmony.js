/*
  Copyright (C) 2012 Ariya Hidayat <ariya.hidayat@gmail.com>
  Copyright (C) 2012 Joost-Wim Boekesteijn <joost-wim@boekesteijn.nl>
  Copyright (C) 2012 Yusuke Suzuki <utatane.tea@gmail.com>
  Copyright (C) 2012 Arpad Borsos <arpad.borsos@googlemail.com>
  Copyright (C) 2011 Ariya Hidayat <ariya.hidayat@gmail.com>
  Copyright (C) 2011 Yusuke Suzuki <utatane.tea@gmail.com>
  Copyright (C) 2011 Arpad Borsos <arpad.borsos@googlemail.com>
  Copyright (C) 2014 Ingvar Stepanyan <me@rreverser.com>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

if (typeof exports != "undefined") {
  var test = require("./driver.js").test;
  var testFail = require("./driver.js").testFail;
  var testAssert = require("./driver.js").testAssert;
}

/*
  Tests below were automatically converted from https://github.com/ariya/esprima/blob/2bb17ef9a45c88e82d72c2c61b7b7af93caef028/test/harmonytest.js.

  Manually fixed locations for:
   - parenthesized expressions (include brackets into expression's location)
   - expression statements (excluded spaces after statement's semicolon)
   - arrow and method functions (included arguments into function's location)
   - template elements (excluded '`', '${' and '}' from element's location)
*/

// ES6 Unicode Code Point Escape Sequence

test("\"\\u{714E}\\u{8336}\"", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: "煎茶",
      raw: "\"\\u{714E}\\u{8336}\"",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("\"\\u{20BB7}\\u{91CE}\\u{5BB6}\"", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: "𠮷野家",
      raw: "\"\\u{20BB7}\\u{91CE}\\u{5BB6}\"",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 27}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: Numeric Literal

test("00", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 0,
      raw: "00",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 2}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 2}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 2}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0o0", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 0,
      raw: "0o0",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function test() {'use strict'; 0o0; }", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "test",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 13}
      }
    },
    params: [],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [
        {
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: "use strict",
            raw: "'use strict'",
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 29}
            }
          },
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 30}
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: 0,
            raw: "0o0",
            loc: {
              start: {line: 1, column: 31},
              end: {line: 1, column: 34}
            }
          },
          loc: {
            start: {line: 1, column: 31},
            end: {line: 1, column: 35}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 16},
        end: {line: 1, column: 37}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 37}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 37}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0o2", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 2,
      raw: "0o2",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0o12", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 10,
      raw: "0o12",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 4}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 4}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 4}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0O0", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 0,
      raw: "0O0",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function test() {'use strict'; 0O0; }", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "test",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 13}
      }
    },
    params: [],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [
        {
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: "use strict",
            raw: "'use strict'",
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 29}
            }
          },
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 30}
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: 0,
            raw: "0O0",
            loc: {
              start: {line: 1, column: 31},
              end: {line: 1, column: 34}
            }
          },
          loc: {
            start: {line: 1, column: 31},
            end: {line: 1, column: 35}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 16},
        end: {line: 1, column: 37}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 37}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 37}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0O2", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 2,
      raw: "0O2",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0O12", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 10,
      raw: "0O12",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 4}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 4}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 4}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0b0", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 0,
      raw: "0b0",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0b1", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 1,
      raw: "0b1",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0b10", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 2,
      raw: "0b10",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 4}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 4}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 4}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0B0", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 0,
      raw: "0B0",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0B1", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 1,
      raw: "0B1",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("0B10", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "Literal",
      value: 2,
      raw: "0B10",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 4}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 4}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 4}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6 Template Strings

test("`42`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TemplateLiteral",
      quasis: [{
        type: "TemplateElement",
        value: {raw: "42", cooked: "42"},
        tail: true,
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 3}
        }
      }],
      expressions: [],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 4}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 4}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 4}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("raw`42`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TaggedTemplateExpression",
      tag: {
        type: "Identifier",
        name: "raw",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 3}
        }
      },
      quasi: {
        type: "TemplateLiteral",
        quasis: [{
          type: "TemplateElement",
          value: {raw: "42", cooked: "42"},
          tail: true,
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 6}
          }
        }],
        expressions: [],
        loc: {
          start: {line: 1, column: 3},
          end: {line: 1, column: 7}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 7}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 7}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 7}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("raw`hello ${name}`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TaggedTemplateExpression",
      tag: {
        type: "Identifier",
        name: "raw",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 3}
        }
      },
      quasi: {
        type: "TemplateLiteral",
        quasis: [
          {
            type: "TemplateElement",
            value: {raw: "hello ", cooked: "hello "},
            tail: false,
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 10}
            }
          },
          {
            type: "TemplateElement",
            value: {raw: "", cooked: ""},
            tail: true,
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 17}
            }
          }
        ],
        expressions: [{
          type: "Identifier",
          name: "name",
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 16}
          }
        }],
        loc: {
          start: {line: 1, column: 3},
          end: {line: 1, column: 18}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("`$`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TemplateLiteral",
      quasis: [{
        type: "TemplateElement",
        value: {raw: "$", cooked: "$"},
        tail: true,
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      expressions: [],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 3}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 3}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("`\\n\\r\\b\\v\\t\\f\\\n\\\r\n`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TemplateLiteral",
      quasis: [{
        type: "TemplateElement",
        value: {raw: "\\n\\r\\b\\v\\t\\f\\\n\\\r\n", cooked: "\n\r\b\u000b\t\f"},
        tail: true,
        loc: {
          start: {line: 1, column: 1},
          end: {line: 3, column: 0}
        }
      }],
      expressions: [],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 3, column: 1}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 3, column: 1}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 3, column: 1}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("`\n\r\n`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TemplateLiteral",
      quasis: [{
        type: "TemplateElement",
        value: {raw: "\n\r\n", cooked: "\n\n"},
        tail: true,
        loc: {
          start: {line: 1, column: 1},
          end: {line: 3, column: 0}
        }
      }],
      expressions: [],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 3, column: 1}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 3, column: 1}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 3, column: 1}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("`\\u{000042}\\u0042\\x42u0\\102\\A`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "TemplateLiteral",
      quasis: [{
        type: "TemplateElement",
        value: {raw: "\\u{000042}\\u0042\\x42u0\\102\\A", cooked: "BBBu0BA"},
        tail: true,
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 29}
        }
      }],
      expressions: [],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 30}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 30}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 30}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("new raw`42`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "NewExpression",
      callee: {
        type: "TaggedTemplateExpression",
        tag: {
          type: "Identifier",
          name: "raw",
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 7}
          }
        },
        quasi: {
          type: "TemplateLiteral",
          quasis: [{
            type: "TemplateElement",
            value: {raw: "42", cooked: "42"},
            tail: true,
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 10}
            }
          }],
          expressions: [],
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 11}
          }
        },
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 11}
        }
      },
      arguments: [],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 11}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 11}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 11}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("`outer${{x: {y: 10}}}bar${`nested${function(){return 1;}}endnest`}end`",{
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "TemplateLiteral",
        expressions: [
          {
            type: "ObjectExpression",
            properties: [
              {
                type: "Property",
                method: false,
                shorthand: false,
                computed: false,
                key: {
                  type: "Identifier",
                  name: "x"
                },
                value: {
                  type: "ObjectExpression",
                  properties: [
                    {
                      type: "Property",
                      method: false,
                      shorthand: false,
                      computed: false,
                      key: {
                        type: "Identifier",
                        name: "y"
                      },
                      value: {
                        type: "Literal",
                        value: 10,
                        raw: "10"
                      },
                      kind: "init"
                    }
                  ]
                },
                kind: "init"
              }
            ]
          },
          {
            type: "TemplateLiteral",
            expressions: [
              {
                type: "FunctionExpression",
                id: null,
                params: [],
                defaults: [],
                rest: null,
                generator: false,
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ReturnStatement",
                      argument: {
                        type: "Literal",
                        value: 1,
                        raw: "1"
                      }
                    }
                  ]
                },
                expression: false
              }
            ],
            quasis: [
              {
                type: "TemplateElement",
                value: {
                  cooked: "nested",
                  raw: "nested"
                },
                tail: false
              },
              {
                type: "TemplateElement",
                value: {
                  cooked: "endnest",
                  raw: "endnest"
                },
                tail: true
              }
            ]
          }
        ],
        quasis: [
          {
            type: "TemplateElement",
            value: {
              cooked: "outer",
              raw: "outer"
            },
            tail: false
          },
          {
            type: "TemplateElement",
            value: {
              cooked: "bar",
              raw: "bar"
            },
            tail: false
          },
          {
            type: "TemplateElement",
            value: {
              cooked: "end",
              raw: "end"
            },
            tail: true
          }
        ]
      }
    }
  ]
}, {
  ecmaVersion: 6
});


// ES6: Switch Case Declaration

test("switch (answer) { case 42: let t = 42; break; }", {
  type: "Program",
  body: [{
    type: "SwitchStatement",
    discriminant: {
      type: "Identifier",
      name: "answer",
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 14}
      }
    },
    cases: [{
      type: "SwitchCase",
      test: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 23},
          end: {line: 1, column: 25}
        }
      },
      consequent: [
        {
          type: "VariableDeclaration",
          declarations: [{
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "t",
              loc: {
                start: {line: 1, column: 31},
                end: {line: 1, column: 32}
              }
            },
            init: {
              type: "Literal",
              value: 42,
              raw: "42",
              loc: {
                start: {line: 1, column: 35},
                end: {line: 1, column: 37}
              }
            },
            loc: {
              start: {line: 1, column: 31},
              end: {line: 1, column: 37}
            }
          }],
          kind: "let",
          loc: {
            start: {line: 1, column: 27},
            end: {line: 1, column: 38}
          }
        },
        {
          type: "BreakStatement",
          label: null,
          loc: {
            start: {line: 1, column: 39},
            end: {line: 1, column: 45}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 18},
        end: {line: 1, column: 45}
      }
    }],
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 47}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 47}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: Arrow Function

test("() => \"test\"", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "Literal",
        value: "test",
        raw: "\"test\"",
        loc: {
          start: {line: 1, column: 6},
          end: {line: 1, column: 12}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("e => \"test\"", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "e",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: "test",
        raw: "\"test\"",
        loc: {
          start: {line: 1, column: 5},
          end: {line: 1, column: 11}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 11}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 11}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 11}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(e) => \"test\"", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "e",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: "test",
        raw: "\"test\"",
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 13}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(a, b) => \"test\"", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 2}
          }
        },
        {
          type: "Identifier",
          name: "b",
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 5}
          }
        }
      ],
      defaults: [],
      body: {
        type: "Literal",
        value: "test",
        raw: "\"test\"",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 16}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 16}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 16}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 16}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("e => { 42; }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "e",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: 42,
            raw: "42",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 9}
            }
          },
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 10}
          }
        }],
        loc: {
          start: {line: 1, column: 5},
          end: {line: 1, column: 12}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("e => ({ property: 42 })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "e",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      }],
      defaults: [],
      body: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "property",
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 16}
            }
          },
          value: {
            type: "Literal",
            value: 42,
            raw: "42",
            loc: {
              start: {line: 1, column: 18},
              end: {line: 1, column: 20}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 8},
            end: {line: 1, column: 20}
          }
        }],
        loc: {
          start: {line: 1, column: 6},
          end: {line: 1, column: 22}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 23}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 23}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 23}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("e => { label: 42 }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "e",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "LabeledStatement",
          label: {
            type: "Identifier",
            name: "label",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 12}
            }
          },
          body: {
            type: "ExpressionStatement",
            expression: {
              type: "Literal",
              value: 42,
              raw: "42",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 16}
              }
            },
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 16}
            }
          },
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 16}
          }
        }],
        loc: {
          start: {line: 1, column: 5},
          end: {line: 1, column: 18}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(a, b) => { 42; }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 2}
          }
        },
        {
          type: "Identifier",
          name: "b",
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 5}
          }
        }
      ],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: 42,
            raw: "42",
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 14}
            }
          },
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 15}
          }
        }],
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 17}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("([a, , b]) => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 2},
              end: {line: 1, column: 3}
            }
          },
          null,
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 9}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 14},
          end: {line: 1, column: 16}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 16}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 16}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 16}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("([a.a]) => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "ArrayPattern",
        elements: [{
          type: "MemberExpression",
          computed: false,
          object: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 2},
              end: {line: 1, column: 3}
            }
          },
          property: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 5}
            }
          },
          loc: {
            start: {line: 1, column: 2},
            end: {line: 1, column: 5}
          }
        }],
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 6}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 13}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(x=1) => x * x", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [{
        type: "Literal",
        value: 1,
        raw: "1",
        loc: {
          start: {line: 1, column: 3},
          end: {line: 1, column: 4}
        }
      }],
      body: {
        type: "BinaryExpression",
        operator: "*",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 10}
          }
        },
        right: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 14}
          }
        },
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 14}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 14}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("eval => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "eval",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 4}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 10}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 10}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 10}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 10}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("arguments => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "arguments",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 9}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 15}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 15}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 15}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 15}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(a) => 00", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "a",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [],
      body: {
        type: "Literal",
        value: 0,
        raw: "00",
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 9}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 9}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 9}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 9}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(eval, a) => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [
        {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 5}
          }
        },
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 8}
          }
        }
      ],
      defaults: [],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 15}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 15}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 15}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 15}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(eval = 10) => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "eval",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 5}
        }
      }],
      defaults: [{
        type: "Literal",
        value: 10,
        raw: "10",
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 10}
        }
      }],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 17}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(eval, a = 10) => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [
        {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 5}
          }
        },
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 8}
          }
        }
      ],
      defaults: [
        null,
        {
          type: "Literal",
          value: 10,
          raw: "10",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 13}
          }
        }
      ],
      body: {
        type: "Literal",
        value: 42,
        raw: "42",
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 20}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 20}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 20}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 20}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(x => x)", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [],
      body: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 6},
          end: {line: 1, column: 7}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 7}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 8}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 8}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x => y => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      }],
      defaults: [],
      body: {
        type: "ArrowFunctionExpression",
        id: null,
        params: [{
          type: "Identifier",
          name: "y",
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 6}
          }
        }],
        defaults: [],
        body: {
          type: "Literal",
          value: 42,
          raw: "42",
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 12}
          }
        },
        rest: null,
        generator: false,
        expression: true,
        loc: {
          start: {line: 1, column: 5},
          end: {line: 1, column: 12}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(x) => ((y, z) => (x, y, z))", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [],
      body: {
        type: "ArrowFunctionExpression",
        id: null,
        params: [
          {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 10}
            }
          },
          {
            type: "Identifier",
            name: "z",
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 13}
            }
          }
        ],
        defaults: [],
        body: {
          type: "SequenceExpression",
          expressions: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {line: 1, column: 19},
                end: {line: 1, column: 20}
              }
            },
            {
              type: "Identifier",
              name: "y",
              loc: {
                start: {line: 1, column: 22},
                end: {line: 1, column: 23}
              }
            },
            {
              type: "Identifier",
              name: "z",
              loc: {
                start: {line: 1, column: 25},
                end: {line: 1, column: 26}
              }
            }
          ],
          loc: {
            start: {line: 1, column: 19},
            end: {line: 1, column: 26}
          }
        },
        rest: null,
        generator: false,
        expression: true,
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 27}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 28}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 28}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 28}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("foo(() => {})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "CallExpression",
      callee: {
        type: "Identifier",
        name: "foo",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 3}
        }
      },
      arguments: [{
        type: "ArrowFunctionExpression",
        id: null,
        params: [],
        defaults: [],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 12}
          }
        },
        rest: null,
        generator: false,
        expression: false,
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 12}
        }
      }],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("foo((x, y) => {})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "CallExpression",
      callee: {
        type: "Identifier",
        name: "foo",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 3}
        }
      },
      arguments: [{
        type: "ArrowFunctionExpression",
        id: null,
        params: [
          {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 9}
            }
          }
        ],
        defaults: [],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {line: 1, column: 14},
            end: {line: 1, column: 16}
          }
        },
        rest: null,
        generator: false,
        expression: false,
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 16}
        }
      }],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(a, a) => 42", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  },
  body: [{
    type: "ExpressionStatement",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    },
    expression: {
      type: "ArrowFunctionExpression",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      },
      id: null,
      params: [
        {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 2}
          },
          name: "a"
        },
        {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 5}
          },
          name: "a"
        }
      ],
      defaults: [],
      rest: null,
      generator: false,
      body: {
        type: "Literal",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 12}
        },
        value: 42,
        raw: "42"
      },
      expression: true
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: Method Definition

test("x = { method() { } }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "method",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 12}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 15},
                end: {line: 1, column: 18}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 18}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 18}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 20}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 20}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 20}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 20}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { method(test) { } }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "method",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 12}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "test",
              loc: {
                start: {line: 1, column: 13},
                end: {line: 1, column: 17}
              }
            }],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 19},
                end: {line: 1, column: 22}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 22}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 22}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 24}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 24}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 24}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 24}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { 'method'() { } }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Literal",
            value: "method",
            raw: "'method'",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 14}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 17},
                end: {line: 1, column: 20}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 20}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 20}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 22}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 22}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { get() { } }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "get",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 9}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 12},
                end: {line: 1, column: 15}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 15}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 15}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 17}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { set() { } }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "set",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 9}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 12},
                end: {line: 1, column: 15}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 15}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 15}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 17}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { method() 42 }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "method",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 12}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "Literal",
              value: 42,
              raw: "42",
              loc: {
                start: {line: 1, column: 15},
                end: {line: 1, column: 17}
              }
            },
            rest: null,
            generator: false,
            expression: true,
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 17}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 17}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 19}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 19}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 19}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 19}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { get method() 42 }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "method",
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 16}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "Literal",
              value: 42,
              raw: "42",
              loc: {
                start: {line: 1, column: 19},
                end: {line: 1, column: 21}
              }
            },
            rest: null,
            generator: false,
            expression: true,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 21}
            }
          },
          kind: "get",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 21}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 23}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 23}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 23}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 23}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { set method(val) v = val }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "method",
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 16}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "val",
              loc: {
                start: {line: 1, column: 17},
                end: {line: 1, column: 20}
              }
            }],
            defaults: [],
            body: {
              type: "AssignmentExpression",
              operator: "=",
              left: {
                type: "Identifier",
                name: "v",
                loc: {
                  start: {line: 1, column: 22},
                  end: {line: 1, column: 23}
                }
              },
              right: {
                type: "Identifier",
                name: "val",
                loc: {
                  start: {line: 1, column: 26},
                  end: {line: 1, column: 29}
                }
              },
              loc: {
                start: {line: 1, column: 22},
                end: {line: 1, column: 29}
              }
            },
            rest: null,
            generator: false,
            expression: true,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 29}
            }
          },
          kind: "set",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 29}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 31}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 31}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 31}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 31}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// Array and Generator Comprehension

test("[for (x of array) x]", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ComprehensionExpression",
      filter: null,
      blocks: [{
        type: "ComprehensionBlock",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 7}
          }
        },
        right: {
          type: "Identifier",
          name: "array",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 16}
          }
        },
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 17}
        },
        of: true
      }],
      body: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 19}
        }
      },
      generator: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 20}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 20}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 20}
  }
}, {
  ecmaVersion: 7,
  ranges: true,
  locations: true
});

test("[for (x of array) for (y of array2) if (x === test) x]", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ComprehensionExpression",
      filter: {
        type: "BinaryExpression",
        operator: "===",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 40},
            end: {line: 1, column: 41}
          }
        },
        right: {
          type: "Identifier",
          name: "test",
          loc: {
            start: {line: 1, column: 46},
            end: {line: 1, column: 50}
          }
        },
        loc: {
          start: {line: 1, column: 40},
          end: {line: 1, column: 50}
        }
      },
      blocks: [
        {
          type: "ComprehensionBlock",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 7}
            }
          },
          right: {
            type: "Identifier",
            name: "array",
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 16}
            }
          },
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 17}
          },
          of: true
        },
        {
          type: "ComprehensionBlock",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 23},
              end: {line: 1, column: 24}
            }
          },
          right: {
            type: "Identifier",
            name: "array2",
            loc: {
              start: {line: 1, column: 28},
              end: {line: 1, column: 34}
            }
          },
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 35}
          },
          of: true
        }
      ],
      body: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 52},
          end: {line: 1, column: 53}
        }
      },
      generator: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 54}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 54}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 54}
  }
}, {
  ecmaVersion: 7,
  ranges: true,
  locations: true
});

test("(for (x of array) for (y of array2) if (x === test) x)", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ComprehensionExpression",
      filter: {
        type: "BinaryExpression",
        operator: "===",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 40},
            end: {line: 1, column: 41}
          }
        },
        right: {
          type: "Identifier",
          name: "test",
          loc: {
            start: {line: 1, column: 46},
            end: {line: 1, column: 50}
          }
        },
        loc: {
          start: {line: 1, column: 40},
          end: {line: 1, column: 50}
        }
      },
      blocks: [
        {
          type: "ComprehensionBlock",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 7}
            }
          },
          right: {
            type: "Identifier",
            name: "array",
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 16}
            }
          },
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 17}
          },
          of: true
        },
        {
          type: "ComprehensionBlock",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 23},
              end: {line: 1, column: 24}
            }
          },
          right: {
            type: "Identifier",
            name: "array2",
            loc: {
              start: {line: 1, column: 28},
              end: {line: 1, column: 34}
            }
          },
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 35}
          },
          of: true
        }
      ],
      body: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 52},
          end: {line: 1, column: 53}
        }
      },
      generator: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 54}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 54}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 54}
  }
}, {
  ecmaVersion: 7,
  ranges: true,
  locations: true
});

test("[for ([,x] of array) for ({[start.x]: x, [start.y]: y} of array2) x]", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ComprehensionExpression",
      filter: null,
      blocks: [
        {
          type: "ComprehensionBlock",
          left: {
            type: "ArrayPattern",
            elements: [
              null,
              {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 8},
                  end: {line: 1, column: 9}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 10}
            }
          },
          right: {
            type: "Identifier",
            name: "array",
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 19}
            }
          },
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 20}
          },
          of: true
        },
        {
          type: "ComprehensionBlock",
          left: {
            type: "ObjectPattern",
            properties: [
              {
                type: "Property",
                key: {
                  type: "MemberExpression",
                  computed: false,
                  object: {
                    type: "Identifier",
                    name: "start",
                    loc: {
                      start: {line: 1, column: 28},
                      end: {line: 1, column: 33}
                    }
                  },
                  property: {
                    type: "Identifier",
                    name: "x",
                    loc: {
                      start: {line: 1, column: 34},
                      end: {line: 1, column: 35}
                    }
                  },
                  loc: {
                    start: {line: 1, column: 28},
                    end: {line: 1, column: 35}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {line: 1, column: 38},
                    end: {line: 1, column: 39}
                  }
                },
                kind: "init",
                method: false,
                shorthand: false,
                computed: true,
                loc: {
                  start: {line: 1, column: 27},
                  end: {line: 1, column: 39}
                }
              },
              {
                type: "Property",
                key: {
                  type: "MemberExpression",
                  computed: false,
                  object: {
                    type: "Identifier",
                    name: "start",
                    loc: {
                      start: {line: 1, column: 42},
                      end: {line: 1, column: 47}
                    }
                  },
                  property: {
                    type: "Identifier",
                    name: "y",
                    loc: {
                      start: {line: 1, column: 48},
                      end: {line: 1, column: 49}
                    }
                  },
                  loc: {
                    start: {line: 1, column: 42},
                    end: {line: 1, column: 49}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "y",
                  loc: {
                    start: {line: 1, column: 52},
                    end: {line: 1, column: 53}
                  }
                },
                kind: "init",
                method: false,
                shorthand: false,
                computed: true,
                loc: {
                  start: {line: 1, column: 41},
                  end: {line: 1, column: 53}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 26},
              end: {line: 1, column: 54}
            }
          },
          right: {
            type: "Identifier",
            name: "array2",
            loc: {
              start: {line: 1, column: 58},
              end: {line: 1, column: 64}
            }
          },
          loc: {
            start: {line: 1, column: 21},
            end: {line: 1, column: 65}
          },
          of: true
        }
      ],
      body: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 66},
          end: {line: 1, column: 67}
        }
      },
      generator: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 68}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 68}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 68}
  }
}, {
  ecmaVersion: 7,
  ranges: true,
  locations: true
});

// Harmony: Object Literal Property Value Shorthand

test("x = { y, z }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "y",
              loc: {
                start: {line: 1, column: 6},
                end: {line: 1, column: 7}
              }
            },
            value: {
              type: "Identifier",
              name: "y",
              loc: {
                start: {line: 1, column: 6},
                end: {line: 1, column: 7}
              }
            },
            kind: "init",
            method: false,
            shorthand: true,
            computed: false,
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 7}
            }
          },
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "z",
              loc: {
                start: {line: 1, column: 9},
                end: {line: 1, column: 10}
              }
            },
            value: {
              type: "Identifier",
              name: "z",
              loc: {
                start: {line: 1, column: 9},
                end: {line: 1, column: 10}
              }
            },
            kind: "init",
            method: false,
            shorthand: true,
            computed: false,
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 10}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// Harmony: Destructuring

test("[a, b] = [b, a]", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 1},
              end: {line: 1, column: 2}
            }
          },
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 5}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 6}
        }
      },
      right: {
        type: "ArrayExpression",
        elements: [
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 11}
            }
          },
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 14}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 15}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 15}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 15}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 15}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ responseText: text }) = res", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "responseText",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 15}
            }
          },
          value: {
            type: "Identifier",
            name: "text",
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 21}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 21}
          }
        }],
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 23}
        }
      },
      right: {
        type: "Identifier",
        name: "res",
        loc: {
          start: {line: 1, column: 27},
          end: {line: 1, column: 30}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 30}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 30}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 30}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("const {a} = {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 8}
          }
        }],
        loc: {
          start: {line: 1, column: 6},
          end: {line: 1, column: 9}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 14}
        }
      },
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 14}
      }
    }],
    kind: "const",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("const [a] = []", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [{
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 8}
          }
        }],
        loc: {
          start: {line: 1, column: 6},
          end: {line: 1, column: 9}
        }
      },
      init: {
        type: "ArrayExpression",
        elements: [],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 14}
        }
      },
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 14}
      }
    }],
    kind: "const",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("let {a} = {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 6}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 7}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [],
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 12}
      }
    }],
    kind: "let",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("let [a] = []", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [{
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 6}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 7}
        }
      },
      init: {
        type: "ArrayExpression",
        elements: [],
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 12}
      }
    }],
    kind: "let",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var {a} = {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 6}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 7}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [],
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 12}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var [a] = []", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [{
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 6}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 7}
        }
      },
      init: {
        type: "ArrayExpression",
        elements: [],
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 12}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("const {a:b} = {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          },
          value: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 10}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 10}
          }
        }],
        loc: {
          start: {line: 1, column: 6},
          end: {line: 1, column: 11}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [],
        loc: {
          start: {line: 1, column: 14},
          end: {line: 1, column: 16}
        }
      },
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 16}
      }
    }],
    kind: "const",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 16}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 16}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("let {a:b} = {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          value: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 8}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 9}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 14}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 14}
      }
    }],
    kind: "let",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var {a:b} = {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          value: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 8}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 9}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 14}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 14}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// Harmony: Modules

test("export var document", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "document",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 19}
          }
        },
        init: null,
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 19}
        }
      }],
      kind: "var",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 19}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 19}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 19}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export var document = { }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "document",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 19}
          }
        },
        init: {
          type: "ObjectExpression",
          properties: [],
          loc: {
            start: {line: 1, column: 22},
            end: {line: 1, column: 25}
          }
        },
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 25}
        }
      }],
      kind: "var",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 25}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 25}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 25}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export let document", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "document",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 19}
          }
        },
        init: null,
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 19}
        }
      }],
      kind: "let",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 19}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 19}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 19}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export let document = { }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "document",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 19}
          }
        },
        init: {
          type: "ObjectExpression",
          properties: [],
          loc: {
            start: {line: 1, column: 22},
            end: {line: 1, column: 25}
          }
        },
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 25}
        }
      }],
      kind: "let",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 25}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 25}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 25}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export const document = { }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "document",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 21}
          }
        },
        init: {
          type: "ObjectExpression",
          properties: [],
          loc: {
            start: {line: 1, column: 24},
            end: {line: 1, column: 27}
          }
        },
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 27}
        }
      }],
      kind: "const",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 27}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export function parse() { }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "parse",
        loc: {
          start: {line: 1, column: 16},
          end: {line: 1, column: 21}
        }
      },
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 24},
          end: {line: 1, column: 27}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 27}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export class Class {}", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "ClassDeclaration",
      id: {
        type: "Identifier",
        name: "Class",
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 18}
        }
      },
      superClass: null,
      body: {
        type: "ClassBody",
        body: [],
        loc: {
          start: {line: 1, column: 19},
          end: {line: 1, column: 21}
        }
      },
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 21}
      }
    },
    default: false,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 21}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 21}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export default 42", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: {
      type: "Literal",
      value: 42,
      raw: "42",
      loc: {
        start: {line: 1, column: 15},
        end: {line: 1, column: 17}
      }
    },
    default: true,
    specifiers: null,
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

testFail("export *", "Unexpected token (1:8)", {ecmaVersion: 6});

test("export * from \"crypto\"", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: null,
    specifiers: [{
      type: "ExportBatchSpecifier",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 8}
      }
    }],
    source: {
      type: "Literal",
      value: "crypto",
      raw: "\"crypto\"",
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 22}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export { encrypt }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: null,
    specifiers: [{
      type: "ExportSpecifier",
      id: {
        type: "Identifier",
        name: "encrypt",
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      },
      name: null,
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 16}
      }
    }],
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export { encrypt, decrypt }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: null,
    specifiers: [
      {
        type: "ExportSpecifier",
        id: {
          type: "Identifier",
          name: "encrypt",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 16}
          }
        },
        name: null,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      },
      {
        type: "ExportSpecifier",
        id: {
          type: "Identifier",
          name: "decrypt",
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 25}
          }
        },
        name: null,
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 25}
        }
      }
    ],
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export { encrypt as default }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: null,
    specifiers: [{
      type: "ExportSpecifier",
      id: {
        type: "Identifier",
        name: "encrypt",
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      },
      name: {
        type: "Identifier",
        name: "default",
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 27}
        }
      },
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 27}
      }
    }],
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 29}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 29}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export { encrypt, decrypt as dec }", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: null,
    specifiers: [
      {
        type: "ExportSpecifier",
        id: {
          type: "Identifier",
          name: "encrypt",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 16}
          }
        },
        name: null,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      },
      {
        type: "ExportSpecifier",
        id: {
          type: "Identifier",
          name: "decrypt",
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 25}
          }
        },
        name: {
          type: "Identifier",
          name: "dec",
          loc: {
            start: {line: 1, column: 29},
            end: {line: 1, column: 32}
          }
        },
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 32}
        }
      }
    ],
    source: null,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 34}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 34}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("export { default } from \"other\"", {
  type: "Program",
  body: [{
    type: "ExportDeclaration",
    declaration: null,
    specifiers: [
      {
        type: "ExportSpecifier",
        id: {
          type: "Identifier",
          name: "default",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 16}
          }
        },
        name: null,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      }
    ],
    source: {
      type: "Literal",
      loc: {
        start: {
          line: 1,
          column: 24
        },
        end: {
          line: 1,
          column: 31
        }
      },
      value: "other",
      raw: "\"other\""
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 31}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 31}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("import \"jquery\"", {
  type: "Program",
  body: [{
    type: "ImportDeclaration",
    specifiers: [],
    source: {
      type: "Literal",
      value: "jquery",
      raw: "\"jquery\"",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 15}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 15}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 15}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("import $ from \"jquery\"", {
  type: "Program",
  body: [{
    type: "ImportDeclaration",
    specifiers: [{
      type: "ImportSpecifier",
      id: {
        type: "Identifier",
        name: "$",
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 8}
        }
      },
      name: null,
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 8}
      }
    }],
    source: {
      type: "Literal",
      value: "jquery",
      raw: "\"jquery\"",
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 22}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("import { encrypt, decrypt } from \"crypto\"", {
  type: "Program",
  body: [{
    type: "ImportDeclaration",
    specifiers: [
      {
        type: "ImportSpecifier",
        id: {
          type: "Identifier",
          name: "encrypt",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 16}
          }
        },
        name: null,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      },
      {
        type: "ImportSpecifier",
        id: {
          type: "Identifier",
          name: "decrypt",
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 25}
          }
        },
        name: null,
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 25}
        }
      }
    ],
    source: {
      type: "Literal",
      value: "crypto",
      raw: "\"crypto\"",
      loc: {
        start: {line: 1, column: 33},
        end: {line: 1, column: 41}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 41}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 41}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("import { encrypt as enc } from \"crypto\"", {
  type: "Program",
  body: [{
    type: "ImportDeclaration",
    specifiers: [{
      type: "ImportSpecifier",
      id: {
        type: "Identifier",
        name: "encrypt",
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 16}
        }
      },
      name: {
        type: "Identifier",
        name: "enc",
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 23}
        }
      },
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 23}
      }
    }],
    source: {
      type: "Literal",
      value: "crypto",
      raw: "\"crypto\"",
      loc: {
        start: {line: 1, column: 31},
        end: {line: 1, column: 39}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 39}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 39}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("import crypto, { decrypt, encrypt as enc } from \"crypto\"", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 56}
  },
  body: [{
    type: "ImportDeclaration",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 56}
    },
    specifiers: [
      {
        type: "ImportSpecifier",
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 13}
        },
        id: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 13}
          },
          name: "crypto"
        },
        name: null,
        default: true
      },
      {
        type: "ImportSpecifier",
        loc: {
          start: {line: 1, column: 17},
          end: {line: 1, column: 24}
        },
        id: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 24}
          },
          name: "decrypt"
        },
        name: null,
        default: false
      },
      {
        type: "ImportSpecifier",
        loc: {
          start: {line: 1, column: 26},
          end: {line: 1, column: 40}
        },
        id: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 26},
            end: {line: 1, column: 33}
          },
          name: "encrypt"
        },
        name: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 37},
            end: {line: 1, column: 40}
          },
          name: "enc"
        },
        default: false
      }
    ],
    source: {
      type: "Literal",
      loc: {
        start: {line: 1, column: 48},
        end: {line: 1, column: 56}
      },
      value: "crypto",
      raw: "\"crypto\""
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

testFail("import default from \"foo\"", "Unexpected token (1:7)", {ecmaVersion: 6});

test("import { null as nil } from \"bar\"", {
  type: "Program",
  body: [{
    type: "ImportDeclaration",
    specifiers: [{
      type: "ImportSpecifier",
      id: {
        type: "Identifier",
        name: "null",
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 13}
        }
      },
      name: {
        type: "Identifier",
        name: "nil",
        loc: {
          start: {line: 1, column: 17},
          end: {line: 1, column: 20}
        }
      },
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 20}
      }
    }],
    source: {
      type: "Literal",
      value: "bar",
      raw: "\"bar\"",
      loc: {
        start: {line: 1, column: 28},
        end: {line: 1, column: 33}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("import * as crypto from \"crypto\"", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 32}
  },
  body: [{
    type: "ImportDeclaration",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 32}
    },
    specifiers: [{
      type: "ImportBatchSpecifier",
      loc: {
        start: {line: 1, column: 7},
        end: {line: 1, column: 18}
      },
      name: {
        type: "Identifier",
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 18}
        },
        name: "crypto"
      }
    }],
    source: {
      type: "Literal",
      loc: {
        start: {line: 1, column: 24},
        end: {line: 1, column: 32}
      },
      value: "crypto",
      raw: "\"crypto\""
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// Harmony: Yield Expression

test("(function* () { yield v })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "ExpressionStatement",
          expression: {
            type: "YieldExpression",
            argument: {
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 22},
                end: {line: 1, column: 23}
              }
            },
            delegate: false,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 23}
            }
          },
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 23}
          }
        }],
        loc: {
          start: {line: 1, column: 14},
          end: {line: 1, column: 25}
        }
      },
      rest: null,
      generator: true,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 25}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 26}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 26}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function* () { yield\nv })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "YieldExpression",
              argument: null,
              delegate: false,
              loc: {
                start: {line: 1, column: 16},
                end: {line: 1, column: 21}
              }
            },
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 21}
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 2, column: 0},
                end: {line: 2, column: 1}
              }
            },
            loc: {
              start: {line: 2, column: 0},
              end: {line: 2, column: 1}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 14},
          end: {line: 2, column: 3}
        }
      },
      rest: null,
      generator: true,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 2, column: 3}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 2, column: 4}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 2, column: 4}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function* () { yield *v })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "ExpressionStatement",
          expression: {
            type: "YieldExpression",
            argument: {
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 23},
                end: {line: 1, column: 24}
              }
            },
            delegate: true,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 24}
            }
          },
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 24}
          }
        }],
        loc: {
          start: {line: 1, column: 14},
          end: {line: 1, column: 26}
        }
      },
      rest: null,
      generator: true,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 26}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function* test () { yield *v }", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "test",
      loc: {
        start: {line: 1, column: 10},
        end: {line: 1, column: 14}
      }
    },
    params: [],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [{
        type: "ExpressionStatement",
        expression: {
          type: "YieldExpression",
          argument: {
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 27},
              end: {line: 1, column: 28}
            }
          },
          delegate: true,
          loc: {
            start: {line: 1, column: 20},
            end: {line: 1, column: 28}
          }
        },
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 28}
        }
      }],
      loc: {
        start: {line: 1, column: 18},
        end: {line: 1, column: 30}
      }
    },
    rest: null,
    generator: true,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 30}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 30}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var x = { *test () { yield *v } };", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 5}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "test",
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 15}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [{
                type: "ExpressionStatement",
                expression: {
                  type: "YieldExpression",
                  argument: {
                    type: "Identifier",
                    name: "v",
                    loc: {
                      start: {line: 1, column: 28},
                      end: {line: 1, column: 29}
                    }
                  },
                  delegate: true,
                  loc: {
                    start: {line: 1, column: 21},
                    end: {line: 1, column: 29}
                  }
                },
                loc: {
                  start: {line: 1, column: 21},
                  end: {line: 1, column: 29}
                }
              }],
              loc: {
                start: {line: 1, column: 19},
                end: {line: 1, column: 31}
              }
            },
            rest: null,
            generator: true,
            expression: false,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 31}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 31}
          }
        }],
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 33}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 33}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 34}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 34}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function* t() {}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "t",
      loc: {
        start: {line: 1, column: 10},
        end: {line: 1, column: 11}
      }
    },
    params: [],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 16}
      }
    },
    rest: null,
    generator: true,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 16}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 16}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function* () { yield yield 10 })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "ExpressionStatement",
          expression: {
            type: "YieldExpression",
            argument: {
              type: "YieldExpression",
              argument: {
                type: "Literal",
                value: 10,
                raw: "10",
                loc: {
                  start: {line: 1, column: 28},
                  end: {line: 1, column: 30}
                }
              },
              delegate: false,
              loc: {
                start: {line: 1, column: 22},
                end: {line: 1, column: 30}
              }
            },
            delegate: false,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 30}
            }
          },
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 30}
          }
        }],
        loc: {
          start: {line: 1, column: 14},
          end: {line: 1, column: 32}
        }
      },
      rest: null,
      generator: true,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 32}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// Harmony: Iterators

test("for(x of list) process(x);", {
  type: "Program",
  body: [{
    type: "ForOfStatement",
    left: {
      type: "Identifier",
      name: "x",
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 5}
      }
    },
    right: {
      type: "Identifier",
      name: "list",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 13}
      }
    },
    body: {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "Identifier",
          name: "process",
          loc: {
            start: {line: 1, column: 15},
            end: {line: 1, column: 22}
          }
        },
        arguments: [{
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 23},
            end: {line: 1, column: 24}
          }
        }],
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 25}
        }
      },
      loc: {
        start: {line: 1, column: 15},
        end: {line: 1, column: 26}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 26}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 26}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("for (var x of list) process(x);", {
  type: "Program",
  body: [{
    type: "ForOfStatement",
    left: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 10}
          }
        },
        init: null,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 10}
        }
      }],
      kind: "var",
      loc: {
        start: {line: 1, column: 5},
        end: {line: 1, column: 10}
      }
    },
    right: {
      type: "Identifier",
      name: "list",
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 18}
      }
    },
    body: {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "Identifier",
          name: "process",
          loc: {
            start: {line: 1, column: 20},
            end: {line: 1, column: 27}
          }
        },
        arguments: [{
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 28},
            end: {line: 1, column: 29}
          }
        }],
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 30}
        }
      },
      loc: {
        start: {line: 1, column: 20},
        end: {line: 1, column: 31}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 31}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 31}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("for (var x = 42 of list) process(x);", {
  type: "Program",
  body: [{
    type: "ForOfStatement",
    left: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 10}
          }
        },
        init: {
          type: "Literal",
          value: 42,
          raw: "42",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 15}
          }
        },
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 15}
        }
      }],
      kind: "var",
      loc: {
        start: {line: 1, column: 5},
        end: {line: 1, column: 15}
      }
    },
    right: {
      type: "Identifier",
      name: "list",
      loc: {
        start: {line: 1, column: 19},
        end: {line: 1, column: 23}
      }
    },
    body: {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "Identifier",
          name: "process",
          loc: {
            start: {line: 1, column: 25},
            end: {line: 1, column: 32}
          }
        },
        arguments: [{
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 33},
            end: {line: 1, column: 34}
          }
        }],
        loc: {
          start: {line: 1, column: 25},
          end: {line: 1, column: 35}
        }
      },
      loc: {
        start: {line: 1, column: 25},
        end: {line: 1, column: 36}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 36}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 36}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("for (let x of list) process(x);", {
  type: "Program",
  body: [{
    type: "ForOfStatement",
    left: {
      type: "VariableDeclaration",
      declarations: [{
        type: "VariableDeclarator",
        id: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 10}
          }
        },
        init: null,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 10}
        }
      }],
      kind: "let",
      loc: {
        start: {line: 1, column: 5},
        end: {line: 1, column: 10}
      }
    },
    right: {
      type: "Identifier",
      name: "list",
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 18}
      }
    },
    body: {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "Identifier",
          name: "process",
          loc: {
            start: {line: 1, column: 20},
            end: {line: 1, column: 27}
          }
        },
        arguments: [{
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 28},
            end: {line: 1, column: 29}
          }
        }],
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 30}
        }
      },
      loc: {
        start: {line: 1, column: 20},
        end: {line: 1, column: 31}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 31}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 31}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// Harmony: Class (strawman)

test("var A = class extends B {}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "Identifier",
        name: "A",
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 5}
        }
      },
      init: {
        type: "ClassExpression",
        superClass: {
          type: "Identifier",
          name: "B",
          loc: {
            start: {line: 1, column: 22},
            end: {line: 1, column: 23}
          }
        },
        body: {
          type: "ClassBody",
          body: [],
          loc: {
            start: {line: 1, column: 24},
            end: {line: 1, column: 26}
          }
        },
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 26}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 26}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 26}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 26}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A extends class B extends C {} {}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: {
      type: "ClassExpression",
      id: {
        type: "Identifier",
        name: "B",
        loc: {
          start: {line: 1, column: 22},
          end: {line: 1, column: 23}
        }
      },
      superClass: {
        type: "Identifier",
        name: "C",
        loc: {
          start: {line: 1, column: 32},
          end: {line: 1, column: 33}
        }
      },
      body: {
        type: "ClassBody",
        body: [],
        loc: {
          start: {line: 1, column: 34},
          end: {line: 1, column: 36}
        }
      },
      loc: {
        start: {line: 1, column: 16},
        end: {line: 1, column: 36}
      }
    },
    body: {
      type: "ClassBody",
      body: [],
      loc: {
        start: {line: 1, column: 37},
        end: {line: 1, column: 39}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 39}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 39}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {get() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "get",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 12}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 15},
              end: {line: 1, column: 17}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 17}
          }
        },
        kind: "",
        static: false,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 17}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static get() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "get",
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 20}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 23},
              end: {line: 1, column: 25}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 20},
            end: {line: 1, column: 25}
          }
        },
        kind: "",
        static: true,
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 25}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 26}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 26}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 26}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A extends B {get foo() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: {
      type: "Identifier",
      name: "B",
      loc: {
        start: {line: 1, column: 16},
        end: {line: 1, column: 17}
      }
    },
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "foo",
          loc: {
            start: {line: 1, column: 23},
            end: {line: 1, column: 26}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 29},
              end: {line: 1, column: 31}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 26},
            end: {line: 1, column: 31}
          }
        },
        kind: "get",
        static: false,
        loc: {
          start: {line: 1, column: 19},
          end: {line: 1, column: 31}
        }
      }],
      loc: {
        start: {line: 1, column: 18},
        end: {line: 1, column: 32}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 32}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 32}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A extends B { static get foo() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: {
      type: "Identifier",
      name: "B",
      loc: {
        start: {line: 1, column: 16},
        end: {line: 1, column: 17}
      }
    },
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "foo",
          loc: {
            start: {line: 1, column: 31},
            end: {line: 1, column: 34}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 37},
              end: {line: 1, column: 39}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 34},
            end: {line: 1, column: 39}
          }
        },
        kind: "get",
        static: true,
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 39}
        }
      }],
      loc: {
        start: {line: 1, column: 18},
        end: {line: 1, column: 40}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 40}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 40}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {set a(v) {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 14}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 15},
              end: {line: 1, column: 16}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 18},
              end: {line: 1, column: 20}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 14},
            end: {line: 1, column: 20}
          }
        },
        kind: "set",
        static: false,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 20}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 21}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 21}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 21}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static set a(v) {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 21},
            end: {line: 1, column: 22}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 23},
              end: {line: 1, column: 24}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 26},
              end: {line: 1, column: 28}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 22},
            end: {line: 1, column: 28}
          }
        },
        kind: "set",
        static: true,
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 28}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 29}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 29}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 29}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {set(v) {};}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "set",
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 12}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 14}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 18}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 18}
          }
        },
        kind: "",
        static: false,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 18}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 20}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 20}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 20}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static set(v) {};}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "set",
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 20}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 21},
              end: {line: 1, column: 22}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 24},
              end: {line: 1, column: 26}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 20},
            end: {line: 1, column: 26}
          }
        },
        kind: "",
        static: true,
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 26}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 28}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 28}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 28}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {*gen(v) { yield v; }}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "gen",
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 13}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 15}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [{
              type: "ExpressionStatement",
              expression: {
                type: "YieldExpression",
                argument: {
                  type: "Identifier",
                  name: "v",
                  loc: {
                    start: {line: 1, column: 25},
                    end: {line: 1, column: 26}
                  }
                },
                delegate: false,
                loc: {
                  start: {line: 1, column: 19},
                  end: {line: 1, column: 26}
                }
              },
              loc: {
                start: {line: 1, column: 19},
                end: {line: 1, column: 27}
              }
            }],
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 29}
            }
          },
          rest: null,
          generator: true,
          expression: false,
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 29}
          }
        },
        kind: "",
        static: false,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 29}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 30}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 30}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 30}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static *gen(v) { yield v; }}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "gen",
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 21}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "Identifier",
            name: "v",
            loc: {
              start: {line: 1, column: 22},
              end: {line: 1, column: 23}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [{
              type: "ExpressionStatement",
              expression: {
                type: "YieldExpression",
                argument: {
                  type: "Identifier",
                  name: "v",
                  loc: {
                    start: {line: 1, column: 33},
                    end: {line: 1, column: 34}
                  }
                },
                delegate: false,
                loc: {
                  start: {line: 1, column: 27},
                  end: {line: 1, column: 34}
                }
              },
              loc: {
                start: {line: 1, column: 27},
                end: {line: 1, column: 35}
              }
            }],
            loc: {
              start: {line: 1, column: 25},
              end: {line: 1, column: 37}
            }
          },
          rest: null,
          generator: true,
          expression: false,
          loc: {
            start: {line: 1, column: 21},
            end: {line: 1, column: 37}
          }
        },
        kind: "",
        static: true,
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 37}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 38}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 38}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 38}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("\"use strict\"; (class A {constructor() { super() }})", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "use strict",
        raw: "\"use strict\"",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    {
      type: "ExpressionStatement",
      expression: {
        type: "ClassExpression",
        id: {
          type: "Identifier",
          name: "A",
          loc: {
            start: {line: 1, column: 21},
            end: {line: 1, column: 22}
          }
        },
        superClass: null,
        body: {
          type: "ClassBody",
          body: [{
            type: "MethodDefinition",
            computed: false,
            key: {
              type: "Identifier",
              name: "constructor",
              loc: {
                start: {line: 1, column: 24},
                end: {line: 1, column: 35}
              }
            },
            value: {
              type: "FunctionExpression",
              id: null,
              params: [],
              defaults: [],
              body: {
                type: "BlockStatement",
                body: [{
                  type: "ExpressionStatement",
                  expression: {
                    type: "CallExpression",
                    callee: {
                      type: "Identifier",
                      name: "super",
                      loc: {
                        start: {line: 1, column: 40},
                        end: {line: 1, column: 45}
                      }
                    },
                    arguments: [],
                    loc: {
                      start: {line: 1, column: 40},
                      end: {line: 1, column: 47}
                    }
                  },
                  loc: {
                    start: {line: 1, column: 40},
                    end: {line: 1, column: 47}
                  }
                }],
                loc: {
                  start: {line: 1, column: 38},
                  end: {line: 1, column: 49}
                }
              },
              rest: null,
              generator: false,
              expression: false,
              loc: {
                start: {line: 1, column: 35},
                end: {line: 1, column: 49}
              }
            },
            kind: "",
            static: false,
            loc: {
              start: {line: 1, column: 24},
              end: {line: 1, column: 49}
            }
          }],
          loc: {
            start: {line: 1, column: 23},
            end: {line: 1, column: 50}
          }
        },
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 50}
        }
      },
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 51}
      }
    }
  ],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 51}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {static foo() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [{
        type: "MethodDefinition",
        computed: false,
        key: {
          type: "Identifier",
          name: "foo",
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 19}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 22},
              end: {line: 1, column: 24}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 19},
            end: {line: 1, column: 24}
          }
        },
        kind: "",
        static: true,
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 24}
        }
      }],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 25}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 25}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 25}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {foo() {} static bar() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 12}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 15},
                end: {line: 1, column: 17}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 17}
            }
          },
          kind: "",
          static: false,
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 17}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {line: 1, column: 25},
              end: {line: 1, column: 28}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 31},
                end: {line: 1, column: 33}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 28},
              end: {line: 1, column: 33}
            }
          },
          kind: "",
          static: true,
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 33}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 34}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 34}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 34}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("\"use strict\"; (class A { static constructor() { super() }})", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "use strict",
        raw: "\"use strict\"",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 12}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    {
      type: "ExpressionStatement",
      expression: {
        type: "ClassExpression",
        id: {
          type: "Identifier",
          name: "A",
          loc: {
            start: {line: 1, column: 21},
            end: {line: 1, column: 22}
          }
        },
        superClass: null,
        body: {
          type: "ClassBody",
          body: [{
            type: "MethodDefinition",
            computed: false,
            key: {
              type: "Identifier",
              name: "constructor",
              loc: {
                start: {line: 1, column: 32},
                end: {line: 1, column: 43}
              }
            },
            value: {
              type: "FunctionExpression",
              id: null,
              params: [],
              defaults: [],
              body: {
                type: "BlockStatement",
                body: [{
                  type: "ExpressionStatement",
                  expression: {
                    type: "CallExpression",
                    callee: {
                      type: "Identifier",
                      name: "super",
                      loc: {
                        start: {line: 1, column: 48},
                        end: {line: 1, column: 53}
                      }
                    },
                    arguments: [],
                    loc: {
                      start: {line: 1, column: 48},
                      end: {line: 1, column: 55}
                    }
                  },
                  loc: {
                    start: {line: 1, column: 48},
                    end: {line: 1, column: 55}
                  }
                }],
                loc: {
                  start: {line: 1, column: 46},
                  end: {line: 1, column: 57}
                }
              },
              rest: null,
              generator: false,
              expression: false,
              loc: {
                start: {line: 1, column: 43},
                end: {line: 1, column: 57}
              }
            },
            kind: "",
            static: true,
            loc: {
              start: {line: 1, column: 25},
              end: {line: 1, column: 57}
            }
          }],
          loc: {
            start: {line: 1, column: 23},
            end: {line: 1, column: 58}
          }
        },
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 58}
        }
      },
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 59}
      }
    }
  ],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 59}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { foo() {} bar() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 13}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 16},
                end: {line: 1, column: 18}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 18}
            }
          },
          kind: "",
          static: false,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 18}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {line: 1, column: 19},
              end: {line: 1, column: 22}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 25},
                end: {line: 1, column: 27}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 22},
              end: {line: 1, column: 27}
            }
          },
          kind: "",
          static: false,
          loc: {
            start: {line: 1, column: 19},
            end: {line: 1, column: 27}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 28}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 28}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 28}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { get foo() {} set foo(v) {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 17}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 20},
                end: {line: 1, column: 22}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 22}
            }
          },
          kind: "get",
          static: false,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 22}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 27},
              end: {line: 1, column: 30}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 31},
                end: {line: 1, column: 32}
              }
            }],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 34},
                end: {line: 1, column: 36}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 30},
              end: {line: 1, column: 36}
            }
          },
          kind: "set",
          static: false,
          loc: {
            start: {line: 1, column: 23},
            end: {line: 1, column: 36}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 37}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 37}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 37}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static get foo() {} get foo() {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 21},
              end: {line: 1, column: 24}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 27},
                end: {line: 1, column: 29}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 24},
              end: {line: 1, column: 29}
            }
          },
          kind: "get",
          static: true,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 29}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 34},
              end: {line: 1, column: 37}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 40},
                end: {line: 1, column: 42}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 37},
              end: {line: 1, column: 42}
            }
          },
          kind: "get",
          static: false,
          loc: {
            start: {line: 1, column: 30},
            end: {line: 1, column: 42}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 43}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 43}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 43}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static get foo() {} static get bar() {} }", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 21},
              end: {line: 1, column: 24}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 27},
                end: {line: 1, column: 29}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 24},
              end: {line: 1, column: 29}
            }
          },
          kind: "get",
          static: true,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 29}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {line: 1, column: 41},
              end: {line: 1, column: 44}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 47},
                end: {line: 1, column: 49}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 44},
              end: {line: 1, column: 49}
            }
          },
          kind: "get",
          static: true,
          loc: {
            start: {line: 1, column: 30},
            end: {line: 1, column: 49}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 51}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 51}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 51}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static get foo() {} static set foo(v) {} get foo() {} set foo(v) {}}", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 21},
              end: {line: 1, column: 24}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 27},
                end: {line: 1, column: 29}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 24},
              end: {line: 1, column: 29}
            }
          },
          kind: "get",
          static: true,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 29}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 41},
              end: {line: 1, column: 44}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 45},
                end: {line: 1, column: 46}
              }
            }],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 48},
                end: {line: 1, column: 50}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 44},
              end: {line: 1, column: 50}
            }
          },
          kind: "set",
          static: true,
          loc: {
            start: {line: 1, column: 30},
            end: {line: 1, column: 50}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 55},
              end: {line: 1, column: 58}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 61},
                end: {line: 1, column: 63}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 58},
              end: {line: 1, column: 63}
            }
          },
          kind: "get",
          static: false,
          loc: {
            start: {line: 1, column: 51},
            end: {line: 1, column: 63}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 68},
              end: {line: 1, column: 71}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 72},
                end: {line: 1, column: 73}
              }
            }],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 75},
                end: {line: 1, column: 77}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 71},
              end: {line: 1, column: 77}
            }
          },
          kind: "set",
          static: false,
          loc: {
            start: {line: 1, column: 64},
            end: {line: 1, column: 77}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 78}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 78}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 78}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});


test("class A { static [foo]() {} }", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 29}
  },
  body: [{
    type: "ClassDeclaration",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 29}
    },
    id: {
      type: "Identifier",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      },
      name: "A"
    },
    superClass: null,
    body: {
      type: "ClassBody",
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 29}
      },
      body: [{
        type: "MethodDefinition",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 27}
        },
        static: true,
        computed: true,
        key: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 21}
          },
          name: "foo"
        },
        kind: "",
        value: {
          type: "FunctionExpression",
          loc: {
            start: {line: 1, column: 22},
            end: {line: 1, column: 27}
          },
          id: null,
          params: [],
          defaults: [],
          rest: null,
          generator: false,
          body: {
            type: "BlockStatement",
            loc: {
              start: {line: 1, column: 25},
              end: {line: 1, column: 27}
            },
            body: []
          },
          expression: false
        }
      }]
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { static get [foo]() {} }", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  },
  body: [{
    type: "ClassDeclaration",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    },
    id: {
      type: "Identifier",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      },
      range: [
        6,
        7
      ],
      name: "A"
    },
    superClass: null,
    body: {
      type: "ClassBody",
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 33}
      },
      body: [{
        type: "MethodDefinition",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 31}
        },
        static: true,
        computed: true,
        key: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 22},
            end: {line: 1, column: 25}
          },
          name: "foo"
        },
        kind: "get",
        value: {
          type: "FunctionExpression",
          loc: {
            start: {line: 1, column: 26},
            end: {line: 1, column: 31}
          },
          id: null,
          params: [],
          defaults: [],
          rest: null,
          generator: false,
          body: {
            type: "BlockStatement",
            loc: {
              start: {line: 1, column: 29},
              end: {line: 1, column: 31}
            },
            body: []
          },
          expression: false
        }
      }]
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { set foo(v) {} get foo() {} }", {
  type: "Program",
  body: [{
    type: "ClassDeclaration",
    id: {
      type: "Identifier",
      name: "A",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    superClass: null,
    body: {
      type: "ClassBody",
      body: [
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 17}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 18},
                end: {line: 1, column: 19}
              }
            }],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 21},
                end: {line: 1, column: 23}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 23}
            }
          },
          kind: "set",
          static: false,
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 23}
          }
        },
        {
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {line: 1, column: 28},
              end: {line: 1, column: 31}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 34},
                end: {line: 1, column: 36}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 31},
              end: {line: 1, column: 36}
            }
          },
          kind: "get",
          static: false,
          loc: {
            start: {line: 1, column: 24},
            end: {line: 1, column: 36}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 38}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 38}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 38}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A { foo() {} get foo() {} }",{
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  },
  body: [{
    type: "ClassDeclaration",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    },
    id: {
      type: "Identifier",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      },
      name: "A"
    },
    superClass: null,
    body: {
      type: "ClassBody",
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 33}
      },
      body: [
        {
          type: "MethodDefinition",
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 18}
          },
          static: false,
          computed: false,
          key: {
            type: "Identifier",
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 13}
            },
            name: "foo"
          },
          kind: "",
          value: {
            type: "FunctionExpression",
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 18}
            },
            id: null,
            params: [],
            defaults: [],
            rest: null,
            generator: false,
            body: {
              type: "BlockStatement",
              loc: {
                start: {line: 1, column: 16},
                end: {line: 1, column: 18}
              },
              body: []
            },
            expression: false
          }
        },
        {
          type: "MethodDefinition",
          loc: {
            start: {line: 1, column: 19},
            end: {line: 1, column: 31}
          },
          static: false,
          computed: false,
          key: {
            type: "Identifier",
            loc: {
              start: {line: 1, column: 23},
              end: {line: 1, column: 26}
            },
            name: "foo"
          },
          kind: "get",
          value: {
            type: "FunctionExpression",
            loc: {
              start: {line: 1, column: 26},
              end: {line: 1, column: 31}
            },
            id: null,
            params: [],
            defaults: [],
            rest: null,
            generator: false,
            body: {
              type: "BlockStatement",
              loc: {
                start: {line: 1, column: 29},
                end: {line: 1, column: 31}
              },
              body: []
            },
            expression: false
          }
        }
      ]
    }
  }]
},{
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: Computed Properties

test("({[x]: 10})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        },
        value: {
          type: "Literal",
          value: 10,
          raw: "10",
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 9}
          }
        },
        kind: "init",
        method: false,
        shorthand: false,
        computed: true,
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 9}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 10}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 11}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 11}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({[\"x\" + \"y\"]: 10})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "BinaryExpression",
          operator: "+",
          left: {
            type: "Literal",
            value: "x",
            raw: "\"x\"",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 6}
            }
          },
          right: {
            type: "Literal",
            value: "y",
            raw: "\"y\"",
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 12}
            }
          },
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 12}
          }
        },
        value: {
          type: "Literal",
          value: 10,
          raw: "10",
          loc: {
            start: {line: 1, column: 15},
            end: {line: 1, column: 17}
          }
        },
        kind: "init",
        method: false,
        shorthand: false,
        computed: true,
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 17}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 19}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 19}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({[x]: function() {}})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 18},
              end: {line: 1, column: 20}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 7},
            end: {line: 1, column: 20}
          }
        },
        kind: "init",
        method: false,
        shorthand: false,
        computed: true,
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 20}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 21}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({[x]: 10, y: 20})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          value: {
            type: "Literal",
            value: 10,
            raw: "10",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 9}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: true,
          loc: {
            start: {line: 1, column: 2},
            end: {line: 1, column: 9}
          }
        },
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 12}
            }
          },
          value: {
            type: "Literal",
            value: 20,
            raw: "20",
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 16}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 16}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({get [x]() {}, set [x](v) {}})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 8}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 12},
                end: {line: 1, column: 14}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 14}
            }
          },
          kind: "get",
          method: false,
          shorthand: false,
          computed: true,
          loc: {
            start: {line: 1, column: 2},
            end: {line: 1, column: 14}
          }
        },
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 21},
              end: {line: 1, column: 22}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "v",
              loc: {
                start: {line: 1, column: 24},
                end: {line: 1, column: 25}
              }
            }],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 27},
                end: {line: 1, column: 29}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 23},
              end: {line: 1, column: 29}
            }
          },
          kind: "set",
          method: false,
          shorthand: false,
          computed: true,
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 29}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 30}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 31}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 31}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({[x]() {}})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 10}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 10}
          }
        },
        kind: "init",
        method: true,
        shorthand: false,
        computed: true,
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 10}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 11}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var {[x]: y} = {y}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 7}
            }
          },
          value: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 11}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: true,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 11}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 12}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 17}
            }
          },
          value: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 17}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 17}
          }
        }],
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 18}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 18}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function f({[x]: y}) {}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "f",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "ObjectPattern",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 14}
          }
        },
        value: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 18}
          }
        },
        kind: "init",
        method: false,
        shorthand: false,
        computed: true,
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 18}
        }
      }],
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 19}
      }
    }],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 21},
        end: {line: 1, column: 23}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 23}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 23}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var x = {*[test]() { yield *v; }}", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 5}
        }
      },
      init: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "test",
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 15}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [],
            defaults: [],
            body: {
              type: "BlockStatement",
              body: [{
                type: "ExpressionStatement",
                expression: {
                  type: "YieldExpression",
                  argument: {
                    type: "Identifier",
                    name: "v",
                    loc: {
                      start: {line: 1, column: 28},
                      end: {line: 1, column: 29}
                    }
                  },
                  delegate: true,
                  loc: {
                    start: {line: 1, column: 21},
                    end: {line: 1, column: 29}
                  }
                },
                loc: {
                  start: {line: 1, column: 21},
                  end: {line: 1, column: 30}
                }
              }],
              loc: {
                start: {line: 1, column: 19},
                end: {line: 1, column: 32}
              }
            },
            rest: null,
            generator: true,
            expression: false,
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 32}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: true,
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 32}
          }
        }],
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 33}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 33}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("class A {[x]() {}}", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  },
  body: [{
    type: "ClassDeclaration",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    },
    id: {
      type: "Identifier",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      },
      name: "A"
    },
    superClass: null,
    body: {
      type: "ClassBody",
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 18}
      },
      body: [{
        type: "MethodDefinition",
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 17}
        },
        static: false,
        computed: true,
        key: {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 10},
            end: {line: 1, column: 11}
          },
          name: "x"
        },
        kind: "",
        value: {
          type: "FunctionExpression",
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 17}
          },
          id: null,
          params: [],
          defaults: [],
          rest: null,
          generator: false,
          body: {
            type: "BlockStatement",
            loc: {
              start: {line: 1, column: 15},
              end: {line: 1, column: 17}
            },
            body: []
          },
          expression: false
        }
      }]
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

testFail("({[x]})", "Unexpected token (1:5)", {ecmaVersion: 6});

// ES6: Default parameters

test("function f([x] = [1]) {}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "f",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "ArrayPattern",
      elements: [{
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 13}
        }
      }],
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 14}
      }
    }],
    defaults: [{
      type: "ArrayExpression",
      elements: [{
        type: "Literal",
        value: 1,
        raw: "1",
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 19}
        }
      }],
      loc: {
        start: {line: 1, column: 17},
        end: {line: 1, column: 20}
      }
    }],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 22},
        end: {line: 1, column: 24}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 24}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 24}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function f({x} = {x: 10}) {}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "f",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "ObjectPattern",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 13}
          }
        },
        value: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 13}
          }
        },
        kind: "init",
        method: false,
        shorthand: true,
        computed: false,
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 13}
        }
      }],
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 14}
      }
    }],
    defaults: [{
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 18},
            end: {line: 1, column: 19}
          }
        },
        value: {
          type: "Literal",
          value: 10,
          raw: "10",
          loc: {
            start: {line: 1, column: 21},
            end: {line: 1, column: 23}
          }
        },
        kind: "init",
        method: false,
        shorthand: false,
        computed: false,
        loc: {
          start: {line: 1, column: 18},
          end: {line: 1, column: 23}
        }
      }],
      loc: {
        start: {line: 1, column: 17},
        end: {line: 1, column: 24}
      }
    }],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 26},
        end: {line: 1, column: 28}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 28}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 28}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("f = function({x} = {x: 10}) {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "f",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "FunctionExpression",
        id: null,
        params: [{
          type: "ObjectPattern",
          properties: [{
            type: "Property",
            key: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 15}
              }
            },
            value: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 15}
              }
            },
            kind: "init",
            method: false,
            shorthand: true,
            computed: false,
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 15}
            }
          }],
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 16}
          }
        }],
        defaults: [{
          type: "ObjectExpression",
          properties: [{
            type: "Property",
            key: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {line: 1, column: 20},
                end: {line: 1, column: 21}
              }
            },
            value: {
              type: "Literal",
              value: 10,
              raw: "10",
              loc: {
                start: {line: 1, column: 23},
                end: {line: 1, column: 25}
              }
            },
            kind: "init",
            method: false,
            shorthand: false,
            computed: false,
            loc: {
              start: {line: 1, column: 20},
              end: {line: 1, column: 25}
            }
          }],
          loc: {
            start: {line: 1, column: 19},
            end: {line: 1, column: 26}
          }
        }],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {line: 1, column: 28},
            end: {line: 1, column: 30}
          }
        },
        rest: null,
        generator: false,
        expression: false,
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 30}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 30}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 30}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 30}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({f: function({x} = {x: 10}) {}})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "f",
          loc: {
            start: {line: 1, column: 2},
            end: {line: 1, column: 3}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "ObjectPattern",
            properties: [{
              type: "Property",
              key: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 15},
                  end: {line: 1, column: 16}
                }
              },
              value: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 15},
                  end: {line: 1, column: 16}
                }
              },
              kind: "init",
              method: false,
              shorthand: true,
              computed: false,
              loc: {
                start: {line: 1, column: 15},
                end: {line: 1, column: 16}
              }
            }],
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 17}
            }
          }],
          defaults: [{
            type: "ObjectExpression",
            properties: [{
              type: "Property",
              key: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 21},
                  end: {line: 1, column: 22}
                }
              },
              value: {
                type: "Literal",
                value: 10,
                raw: "10",
                loc: {
                  start: {line: 1, column: 24},
                  end: {line: 1, column: 26}
                }
              },
              kind: "init",
              method: false,
              shorthand: false,
              computed: false,
              loc: {
                start: {line: 1, column: 21},
                end: {line: 1, column: 26}
              }
            }],
            loc: {
              start: {line: 1, column: 20},
              end: {line: 1, column: 27}
            }
          }],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 29},
              end: {line: 1, column: 31}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 31}
          }
        },
        kind: "init",
        method: false,
        shorthand: false,
        computed: false,
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 31}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 32}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({f({x} = {x: 10}) {}})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "f",
          loc: {
            start: {line: 1, column: 2},
            end: {line: 1, column: 3}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "ObjectPattern",
            properties: [{
              type: "Property",
              key: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 5},
                  end: {line: 1, column: 6}
                }
              },
              value: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 5},
                  end: {line: 1, column: 6}
                }
              },
              kind: "init",
              method: false,
              shorthand: true,
              computed: false,
              loc: {
                start: {line: 1, column: 5},
                end: {line: 1, column: 6}
              }
            }],
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 7}
            }
          }],
          defaults: [{
            type: "ObjectExpression",
            properties: [{
              type: "Property",
              key: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {line: 1, column: 11},
                  end: {line: 1, column: 12}
                }
              },
              value: {
                type: "Literal",
                value: 10,
                raw: "10",
                loc: {
                  start: {line: 1, column: 14},
                  end: {line: 1, column: 16}
                }
              },
              kind: "init",
              method: false,
              shorthand: false,
              computed: false,
              loc: {
                start: {line: 1, column: 11},
                end: {line: 1, column: 16}
              }
            }],
            loc: {
              start: {line: 1, column: 10},
              end: {line: 1, column: 17}
            }
          }],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 19},
              end: {line: 1, column: 21}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 21}
          }
        },
        kind: "init",
        method: true,
        shorthand: false,
        computed: false,
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 21}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 22}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 23}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 23}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(class {f({x} = {x: 10}) {}})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ClassExpression",
      superClass: null,
      body: {
        type: "ClassBody",
        body: [{
          type: "MethodDefinition",
          computed: false,
          key: {
            type: "Identifier",
            name: "f",
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 9}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "ObjectPattern",
              properties: [{
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {line: 1, column: 11},
                    end: {line: 1, column: 12}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {line: 1, column: 11},
                    end: {line: 1, column: 12}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 11},
                  end: {line: 1, column: 12}
                }
              }],
              loc: {
                start: {line: 1, column: 10},
                end: {line: 1, column: 13}
              }
            }],
            defaults: [{
              type: "ObjectExpression",
              properties: [{
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {line: 1, column: 17},
                    end: {line: 1, column: 18}
                  }
                },
                value: {
                  type: "Literal",
                  value: 10,
                  raw: "10",
                  loc: {
                    start: {line: 1, column: 20},
                    end: {line: 1, column: 22}
                  }
                },
                kind: "init",
                method: false,
                shorthand: false,
                computed: false,
                loc: {
                  start: {line: 1, column: 17},
                  end: {line: 1, column: 22}
                }
              }],
              loc: {
                start: {line: 1, column: 16},
                end: {line: 1, column: 23}
              }
            }],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 25},
                end: {line: 1, column: 27}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 27}
            }
          },
          kind: "",
          static: false,
          loc: {
            start: {line: 1, column: 8},
            end: {line: 1, column: 27}
          }
        }],
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 28}
        }
      },
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 28}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 29}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 29}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(({x} = {x: 10}) => {})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          value: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        }],
        loc: {
          start: {line: 1, column: 2},
          end: {line: 1, column: 5}
        }
      }],
      defaults: [{
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 10}
            }
          },
          value: {
            type: "Literal",
            value: 10,
            raw: "10",
            loc: {
              start: {line: 1, column: 12},
              end: {line: 1, column: 14}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 9},
            end: {line: 1, column: 14}
          }
        }],
        loc: {
          start: {line: 1, column: 8},
          end: {line: 1, column: 15}
        }
      }],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 20},
          end: {line: 1, column: 22}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 22}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 23}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 23}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = function(y = 1) {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "FunctionExpression",
        id: null,
        params: [{
          type: "Identifier",
          name: "y",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 14}
          }
        }],
        defaults: [{
          type: "Literal",
          value: 1,
          raw: "1",
          loc: {
            start: {line: 1, column: 17},
            end: {line: 1, column: 18}
          }
        }],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {line: 1, column: 20},
            end: {line: 1, column: 22}
          }
        },
        rest: null,
        generator: false,
        expression: false,
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 22}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 22}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function f(a = 1) {}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "f",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "Identifier",
      name: "a",
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 12}
      }
    }],
    defaults: [{
      type: "Literal",
      value: 1,
      raw: "1",
      loc: {
        start: {line: 1, column: 15},
        end: {line: 1, column: 16}
      }
    }],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 18},
        end: {line: 1, column: 20}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 20}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 20}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { f: function(a=1) {} }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "f",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 7}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "a",
              loc: {
                start: {line: 1, column: 18},
                end: {line: 1, column: 19}
              }
            }],
            defaults: [{
              type: "Literal",
              value: 1,
              raw: "1",
              loc: {
                start: {line: 1, column: 20},
                end: {line: 1, column: 21}
              }
            }],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 23},
                end: {line: 1, column: 25}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 9},
              end: {line: 1, column: 25}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 25}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 27}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 27}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("x = { f(a=1) {} }", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      },
      right: {
        type: "ObjectExpression",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "f",
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 7}
            }
          },
          value: {
            type: "FunctionExpression",
            id: null,
            params: [{
              type: "Identifier",
              name: "a",
              loc: {
                start: {line: 1, column: 8},
                end: {line: 1, column: 9}
              }
            }],
            defaults: [{
              type: "Literal",
              value: 1,
              raw: "1",
              loc: {
                start: {line: 1, column: 10},
                end: {line: 1, column: 11}
              }
            }],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {line: 1, column: 13},
                end: {line: 1, column: 15}
              }
            },
            rest: null,
            generator: false,
            expression: false,
            loc: {
              start: {line: 1, column: 7},
              end: {line: 1, column: 15}
            }
          },
          kind: "init",
          method: true,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 6},
            end: {line: 1, column: 15}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 17}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: Rest parameters

test("function f(a, ...b) {}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "f",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "Identifier",
      name: "a",
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 12}
      }
    }],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 20},
        end: {line: 1, column: 22}
      }
    },
    rest: {
      type: "Identifier",
      name: "b",
      loc: {
        start: {line: 1, column: 17},
        end: {line: 1, column: 18}
      }
    },
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: Destructured Parameters

test("function x([ a, b ]){}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "x",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "ArrayPattern",
      elements: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 14}
          }
        },
        {
          type: "Identifier",
          name: "b",
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 17}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 19}
      }
    }],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 20},
        end: {line: 1, column: 22}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function x({ a, b }){}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "x",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "ObjectPattern",
      properties: [
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 14}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 14}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 14}
          }
        },
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 17}
            }
          },
          value: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 17}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 17}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 19}
      }
    }],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 20},
        end: {line: 1, column: 22}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function x(a, { a }){}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "x",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [
      {
        type: "Identifier",
        name: "a",
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 12}
        }
      },
      {
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 17}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 17}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 17}
          }
        }],
        loc: {
          start: {line: 1, column: 14},
          end: {line: 1, column: 19}
        }
      }
    ],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 20},
        end: {line: 1, column: 22}
      }
    },
    rest: null,
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function x(...[ a, b ]){}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "x",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 23},
        end: {line: 1, column: 25}
      }
    },
    rest: {
      type: "ArrayPattern",
      elements: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 16},
            end: {line: 1, column: 17}
          }
        },
        {
          type: "Identifier",
          name: "b",
          loc: {
            start: {line: 1, column: 19},
            end: {line: 1, column: 20}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 22}
      }
    },
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 25}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 25}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("function x({ a: { w, x }, b: [y, z] }, ...[a, b, c]){}", {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "x",
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    params: [{
      type: "ObjectPattern",
      properties: [
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 13},
              end: {line: 1, column: 14}
            }
          },
          value: {
            type: "ObjectPattern",
            properties: [
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "w",
                  loc: {
                    start: {line: 1, column: 18},
                    end: {line: 1, column: 19}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "w",
                  loc: {
                    start: {line: 1, column: 18},
                    end: {line: 1, column: 19}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 18},
                  end: {line: 1, column: 19}
                }
              },
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {line: 1, column: 21},
                    end: {line: 1, column: 22}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {line: 1, column: 21},
                    end: {line: 1, column: 22}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 21},
                  end: {line: 1, column: 22}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 16},
              end: {line: 1, column: 24}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 13},
            end: {line: 1, column: 24}
          }
        },
        {
          type: "Property",
          key: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 26},
              end: {line: 1, column: 27}
            }
          },
          value: {
            type: "ArrayPattern",
            elements: [
              {
                type: "Identifier",
                name: "y",
                loc: {
                  start: {line: 1, column: 30},
                  end: {line: 1, column: 31}
                }
              },
              {
                type: "Identifier",
                name: "z",
                loc: {
                  start: {line: 1, column: 33},
                  end: {line: 1, column: 34}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 29},
              end: {line: 1, column: 35}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 26},
            end: {line: 1, column: 35}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 37}
      }
    }],
    defaults: [],
    body: {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {line: 1, column: 52},
        end: {line: 1, column: 54}
      }
    },
    rest: {
      type: "ArrayPattern",
      elements: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 43},
            end: {line: 1, column: 44}
          }
        },
        {
          type: "Identifier",
          name: "b",
          loc: {
            start: {line: 1, column: 46},
            end: {line: 1, column: 47}
          }
        },
        {
          type: "Identifier",
          name: "c",
          loc: {
            start: {line: 1, column: 49},
            end: {line: 1, column: 50}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 42},
        end: {line: 1, column: 51}
      }
    },
    generator: false,
    expression: false,
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 54}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 54}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function x([ a, b ]){})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 11}
        }
      },
      params: [{
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 15}
            }
          },
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 18}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 20}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 21},
          end: {line: 1, column: 23}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 23}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 24}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 24}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function x({ a, b }){})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 11}
        }
      },
      params: [{
        type: "ObjectPattern",
        properties: [
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "a",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 15}
              }
            },
            value: {
              type: "Identifier",
              name: "a",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 15}
              }
            },
            kind: "init",
            method: false,
            shorthand: true,
            computed: false,
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 15}
            }
          },
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "b",
              loc: {
                start: {line: 1, column: 17},
                end: {line: 1, column: 18}
              }
            },
            value: {
              type: "Identifier",
              name: "b",
              loc: {
                start: {line: 1, column: 17},
                end: {line: 1, column: 18}
              }
            },
            kind: "init",
            method: false,
            shorthand: true,
            computed: false,
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 18}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 20}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 21},
          end: {line: 1, column: 23}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 23}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 24}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 24}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function x(...[ a, b ]){})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 11}
        }
      },
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 24},
          end: {line: 1, column: 26}
        }
      },
      rest: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 18}
            }
          },
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 20},
              end: {line: 1, column: 21}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 23}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 26}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(function x({ a: { w, x }, b: [y, z] }, ...[a, b, c]){})", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 11}
        }
      },
      params: [{
        type: "ObjectPattern",
        properties: [
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "a",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 15}
              }
            },
            value: {
              type: "ObjectPattern",
              properties: [
                {
                  type: "Property",
                  key: {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {line: 1, column: 19},
                      end: {line: 1, column: 20}
                    }
                  },
                  value: {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {line: 1, column: 19},
                      end: {line: 1, column: 20}
                    }
                  },
                  kind: "init",
                  method: false,
                  shorthand: true,
                  computed: false,
                  loc: {
                    start: {line: 1, column: 19},
                    end: {line: 1, column: 20}
                  }
                },
                {
                  type: "Property",
                  key: {
                    type: "Identifier",
                    name: "x",
                    loc: {
                      start: {line: 1, column: 22},
                      end: {line: 1, column: 23}
                    }
                  },
                  value: {
                    type: "Identifier",
                    name: "x",
                    loc: {
                      start: {line: 1, column: 22},
                      end: {line: 1, column: 23}
                    }
                  },
                  kind: "init",
                  method: false,
                  shorthand: true,
                  computed: false,
                  loc: {
                    start: {line: 1, column: 22},
                    end: {line: 1, column: 23}
                  }
                }
              ],
              loc: {
                start: {line: 1, column: 17},
                end: {line: 1, column: 25}
              }
            },
            kind: "init",
            method: false,
            shorthand: false,
            computed: false,
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 25}
            }
          },
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "b",
              loc: {
                start: {line: 1, column: 27},
                end: {line: 1, column: 28}
              }
            },
            value: {
              type: "ArrayPattern",
              elements: [
                {
                  type: "Identifier",
                  name: "y",
                  loc: {
                    start: {line: 1, column: 31},
                    end: {line: 1, column: 32}
                  }
                },
                {
                  type: "Identifier",
                  name: "z",
                  loc: {
                    start: {line: 1, column: 34},
                    end: {line: 1, column: 35}
                  }
                }
              ],
              loc: {
                start: {line: 1, column: 30},
                end: {line: 1, column: 36}
              }
            },
            kind: "init",
            method: false,
            shorthand: false,
            computed: false,
            loc: {
              start: {line: 1, column: 27},
              end: {line: 1, column: 36}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 38}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 53},
          end: {line: 1, column: 55}
        }
      },
      rest: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 44},
              end: {line: 1, column: 45}
            }
          },
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 47},
              end: {line: 1, column: 48}
            }
          },
          {
            type: "Identifier",
            name: "c",
            loc: {
              start: {line: 1, column: 50},
              end: {line: 1, column: 51}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 43},
          end: {line: 1, column: 52}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 55}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 56}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 56}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ x([ a, b ]){} })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "ArrayPattern",
            elements: [
              {
                type: "Identifier",
                name: "a",
                loc: {
                  start: {line: 1, column: 7},
                  end: {line: 1, column: 8}
                }
              },
              {
                type: "Identifier",
                name: "b",
                loc: {
                  start: {line: 1, column: 10},
                  end: {line: 1, column: 11}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 13}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 14},
              end: {line: 1, column: 16}
            }
          },
          rest: null,
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 16}
          }
        },
        kind: "init",
        method: true,
        shorthand: false,
        computed: false,
        loc: {
          start: {line: 1, column: 3},
          end: {line: 1, column: 16}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 19}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 19}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ x(...[ a, b ]){} })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 17},
              end: {line: 1, column: 19}
            }
          },
          rest: {
            type: "ArrayPattern",
            elements: [
              {
                type: "Identifier",
                name: "a",
                loc: {
                  start: {line: 1, column: 10},
                  end: {line: 1, column: 11}
                }
              },
              {
                type: "Identifier",
                name: "b",
                loc: {
                  start: {line: 1, column: 13},
                  end: {line: 1, column: 14}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 16}
            }
          },
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 19}
          }
        },
        kind: "init",
        method: true,
        shorthand: false,
        computed: false,
        loc: {
          start: {line: 1, column: 3},
          end: {line: 1, column: 19}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 21}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ x({ a: { w, x }, b: [y, z] }, ...[a, b, c]){} })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        },
        value: {
          type: "FunctionExpression",
          id: null,
          params: [{
            type: "ObjectPattern",
            properties: [
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "a",
                  loc: {
                    start: {line: 1, column: 7},
                    end: {line: 1, column: 8}
                  }
                },
                value: {
                  type: "ObjectPattern",
                  properties: [
                    {
                      type: "Property",
                      key: {
                        type: "Identifier",
                        name: "w",
                        loc: {
                          start: {line: 1, column: 12},
                          end: {line: 1, column: 13}
                        }
                      },
                      value: {
                        type: "Identifier",
                        name: "w",
                        loc: {
                          start: {line: 1, column: 12},
                          end: {line: 1, column: 13}
                        }
                      },
                      kind: "init",
                      method: false,
                      shorthand: true,
                      computed: false,
                      loc: {
                        start: {line: 1, column: 12},
                        end: {line: 1, column: 13}
                      }
                    },
                    {
                      type: "Property",
                      key: {
                        type: "Identifier",
                        name: "x",
                        loc: {
                          start: {line: 1, column: 15},
                          end: {line: 1, column: 16}
                        }
                      },
                      value: {
                        type: "Identifier",
                        name: "x",
                        loc: {
                          start: {line: 1, column: 15},
                          end: {line: 1, column: 16}
                        }
                      },
                      kind: "init",
                      method: false,
                      shorthand: true,
                      computed: false,
                      loc: {
                        start: {line: 1, column: 15},
                        end: {line: 1, column: 16}
                      }
                    }
                  ],
                  loc: {
                    start: {line: 1, column: 10},
                    end: {line: 1, column: 18}
                  }
                },
                kind: "init",
                method: false,
                shorthand: false,
                computed: false,
                loc: {
                  start: {line: 1, column: 7},
                  end: {line: 1, column: 18}
                }
              },
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 20},
                    end: {line: 1, column: 21}
                  }
                },
                value: {
                  type: "ArrayPattern",
                  elements: [
                    {
                      type: "Identifier",
                      name: "y",
                      loc: {
                        start: {line: 1, column: 24},
                        end: {line: 1, column: 25}
                      }
                    },
                    {
                      type: "Identifier",
                      name: "z",
                      loc: {
                        start: {line: 1, column: 27},
                        end: {line: 1, column: 28}
                      }
                    }
                  ],
                  loc: {
                    start: {line: 1, column: 23},
                    end: {line: 1, column: 29}
                  }
                },
                kind: "init",
                method: false,
                shorthand: false,
                computed: false,
                loc: {
                  start: {line: 1, column: 20},
                  end: {line: 1, column: 29}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 31}
            }
          }],
          defaults: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {line: 1, column: 46},
              end: {line: 1, column: 48}
            }
          },
          rest: {
            type: "ArrayPattern",
            elements: [
              {
                type: "Identifier",
                name: "a",
                loc: {
                  start: {line: 1, column: 37},
                  end: {line: 1, column: 38}
                }
              },
              {
                type: "Identifier",
                name: "b",
                loc: {
                  start: {line: 1, column: 40},
                  end: {line: 1, column: 41}
                }
              },
              {
                type: "Identifier",
                name: "c",
                loc: {
                  start: {line: 1, column: 43},
                  end: {line: 1, column: 44}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 36},
              end: {line: 1, column: 45}
            }
          },
          generator: false,
          expression: false,
          loc: {
            start: {line: 1, column: 4},
            end: {line: 1, column: 48}
          }
        },
        kind: "init",
        method: true,
        shorthand: false,
        computed: false,
        loc: {
          start: {line: 1, column: 3},
          end: {line: 1, column: 48}
        }
      }],
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 50}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 51}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 51}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(...a) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 10},
          end: {line: 1, column: 12}
        }
      },
      rest: {
        type: "Identifier",
        name: "a",
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 5}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(a, ...b) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "a",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 15}
        }
      },
      rest: {
        type: "Identifier",
        name: "b",
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 8}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 15}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 15}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 15}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ a }) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        }],
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 6}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 13}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ a }, ...b) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          value: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          kind: "init",
          method: false,
          shorthand: true,
          computed: false,
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 4}
          }
        }],
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 6}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 17},
          end: {line: 1, column: 19}
        }
      },
      rest: {
        type: "Identifier",
        name: "b",
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 12}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 19}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 19}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 19}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(...[a, b]) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 17}
        }
      },
      rest: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 9}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 10}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("(a, ...[b]) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "a",
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 2}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 15},
          end: {line: 1, column: 17}
        }
      },
      rest: {
        type: "ArrayPattern",
        elements: [{
          type: "Identifier",
          name: "b",
          loc: {
            start: {line: 1, column: 8},
            end: {line: 1, column: 9}
          }
        }],
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 10}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 17}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ a: [a, b] }, ...c) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "ObjectPattern",
        properties: [{
          type: "Property",
          key: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 3},
              end: {line: 1, column: 4}
            }
          },
          value: {
            type: "ArrayPattern",
            elements: [
              {
                type: "Identifier",
                name: "a",
                loc: {
                  start: {line: 1, column: 7},
                  end: {line: 1, column: 8}
                }
              },
              {
                type: "Identifier",
                name: "b",
                loc: {
                  start: {line: 1, column: 10},
                  end: {line: 1, column: 11}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 6},
              end: {line: 1, column: 12}
            }
          },
          kind: "init",
          method: false,
          shorthand: false,
          computed: false,
          loc: {
            start: {line: 1, column: 3},
            end: {line: 1, column: 12}
          }
        }],
        loc: {
          start: {line: 1, column: 1},
          end: {line: 1, column: 14}
        }
      }],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 25},
          end: {line: 1, column: 27}
        }
      },
      rest: {
        type: "Identifier",
        name: "c",
        loc: {
          start: {line: 1, column: 19},
          end: {line: 1, column: 20}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 27}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("({ a: b, c }, [d, e], ...f) => {}", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [
        {
          type: "ObjectPattern",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "a",
                loc: {
                  start: {line: 1, column: 3},
                  end: {line: 1, column: 4}
                }
              },
              value: {
                type: "Identifier",
                name: "b",
                loc: {
                  start: {line: 1, column: 6},
                  end: {line: 1, column: 7}
                }
              },
              kind: "init",
              method: false,
              shorthand: false,
              computed: false,
              loc: {
                start: {line: 1, column: 3},
                end: {line: 1, column: 7}
              }
            },
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "c",
                loc: {
                  start: {line: 1, column: 9},
                  end: {line: 1, column: 10}
                }
              },
              value: {
                type: "Identifier",
                name: "c",
                loc: {
                  start: {line: 1, column: 9},
                  end: {line: 1, column: 10}
                }
              },
              kind: "init",
              method: false,
              shorthand: true,
              computed: false,
              loc: {
                start: {line: 1, column: 9},
                end: {line: 1, column: 10}
              }
            }
          ],
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 12}
          }
        },
        {
          type: "ArrayPattern",
          elements: [
            {
              type: "Identifier",
              name: "d",
              loc: {
                start: {line: 1, column: 15},
                end: {line: 1, column: 16}
              }
            },
            {
              type: "Identifier",
              name: "e",
              loc: {
                start: {line: 1, column: 18},
                end: {line: 1, column: 19}
              }
            }
          ],
          loc: {
            start: {line: 1, column: 14},
            end: {line: 1, column: 20}
          }
        }
      ],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {line: 1, column: 31},
          end: {line: 1, column: 33}
        }
      },
      rest: {
        type: "Identifier",
        name: "f",
        loc: {
          start: {line: 1, column: 25},
          end: {line: 1, column: 26}
        }
      },
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 33}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 33}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 33}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

// ES6: SpreadElement

test("[...a] = b", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "ArrayPattern",
        elements: [{
          type: "SpreadElement",
          argument: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 5}
            }
          },
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 5}
          }
        }],
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 6}
        }
      },
      right: {
        type: "Identifier",
        name: "b",
        loc: {
          start: {line: 1, column: 9},
          end: {line: 1, column: 10}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 10}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 10}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 10}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("[a, ...b] = c", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 1},
              end: {line: 1, column: 2}
            }
          },
          {
            type: "SpreadElement",
            argument: {
              type: "Identifier",
              name: "b",
              loc: {
                start: {line: 1, column: 7},
                end: {line: 1, column: 8}
              }
            },
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 8}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 9}
        }
      },
      right: {
        type: "Identifier",
        name: "c",
        loc: {
          start: {line: 1, column: 12},
          end: {line: 1, column: 13}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("[{ a, b }, ...c] = d", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "ArrayPattern",
        elements: [
          {
            type: "ObjectPattern",
            properties: [
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "a",
                  loc: {
                    start: {line: 1, column: 3},
                    end: {line: 1, column: 4}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "a",
                  loc: {
                    start: {line: 1, column: 3},
                    end: {line: 1, column: 4}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 3},
                  end: {line: 1, column: 4}
                }
              },
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 6},
                    end: {line: 1, column: 7}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 6},
                    end: {line: 1, column: 7}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 6},
                  end: {line: 1, column: 7}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 1},
              end: {line: 1, column: 9}
            }
          },
          {
            type: "SpreadElement",
            argument: {
              type: "Identifier",
              name: "c",
              loc: {
                start: {line: 1, column: 14},
                end: {line: 1, column: 15}
              }
            },
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 15}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 16}
        }
      },
      right: {
        type: "Identifier",
        name: "d",
        loc: {
          start: {line: 1, column: 19},
          end: {line: 1, column: 20}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 20}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 20}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 20}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("[a, ...[b, c]] = d", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "AssignmentExpression",
      operator: "=",
      left: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 1},
              end: {line: 1, column: 2}
            }
          },
          {
            type: "SpreadElement",
            argument: {
              type: "ArrayPattern",
              elements: [
                {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 8},
                    end: {line: 1, column: 9}
                  }
                },
                {
                  type: "Identifier",
                  name: "c",
                  loc: {
                    start: {line: 1, column: 11},
                    end: {line: 1, column: 12}
                  }
                }
              ],
              loc: {
                start: {line: 1, column: 7},
                end: {line: 1, column: 13}
              }
            },
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 13}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 14}
        }
      },
      right: {
        type: "Identifier",
        name: "d",
        loc: {
          start: {line: 1, column: 17},
          end: {line: 1, column: 18}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 18}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 18}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 18}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var [...a] = b", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [{
          type: "SpreadElement",
          argument: {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 9}
            }
          },
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 9}
          }
        }],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 10}
        }
      },
      init: {
        type: "Identifier",
        name: "b",
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 14}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 14}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var [a, ...b] = c", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          {
            type: "SpreadElement",
            argument: {
              type: "Identifier",
              name: "b",
              loc: {
                start: {line: 1, column: 11},
                end: {line: 1, column: 12}
              }
            },
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 12}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 13}
        }
      },
      init: {
        type: "Identifier",
        name: "c",
        loc: {
          start: {line: 1, column: 16},
          end: {line: 1, column: 17}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 17}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 17}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 17}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var [{ a, b }, ...c] = d", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [
          {
            type: "ObjectPattern",
            properties: [
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "a",
                  loc: {
                    start: {line: 1, column: 7},
                    end: {line: 1, column: 8}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "a",
                  loc: {
                    start: {line: 1, column: 7},
                    end: {line: 1, column: 8}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 7},
                  end: {line: 1, column: 8}
                }
              },
              {
                type: "Property",
                key: {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 10},
                    end: {line: 1, column: 11}
                  }
                },
                value: {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 10},
                    end: {line: 1, column: 11}
                  }
                },
                kind: "init",
                method: false,
                shorthand: true,
                computed: false,
                loc: {
                  start: {line: 1, column: 10},
                  end: {line: 1, column: 11}
                }
              }
            ],
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 13}
            }
          },
          {
            type: "SpreadElement",
            argument: {
              type: "Identifier",
              name: "c",
              loc: {
                start: {line: 1, column: 18},
                end: {line: 1, column: 19}
              }
            },
            loc: {
              start: {line: 1, column: 15},
              end: {line: 1, column: 19}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 20}
        }
      },
      init: {
        type: "Identifier",
        name: "d",
        loc: {
          start: {line: 1, column: 23},
          end: {line: 1, column: 24}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 24}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 24}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 24}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("var [a, ...[b, c]] = d", {
  type: "Program",
  body: [{
    type: "VariableDeclaration",
    declarations: [{
      type: "VariableDeclarator",
      id: {
        type: "ArrayPattern",
        elements: [
          {
            type: "Identifier",
            name: "a",
            loc: {
              start: {line: 1, column: 5},
              end: {line: 1, column: 6}
            }
          },
          {
            type: "SpreadElement",
            argument: {
              type: "ArrayPattern",
              elements: [
                {
                  type: "Identifier",
                  name: "b",
                  loc: {
                    start: {line: 1, column: 12},
                    end: {line: 1, column: 13}
                  }
                },
                {
                  type: "Identifier",
                  name: "c",
                  loc: {
                    start: {line: 1, column: 15},
                    end: {line: 1, column: 16}
                  }
                }
              ],
              loc: {
                start: {line: 1, column: 11},
                end: {line: 1, column: 17}
              }
            },
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 17}
            }
          }
        ],
        loc: {
          start: {line: 1, column: 4},
          end: {line: 1, column: 18}
        }
      },
      init: {
        type: "Identifier",
        name: "d",
        loc: {
          start: {line: 1, column: 21},
          end: {line: 1, column: 22}
        }
      },
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 22}
      }
    }],
    kind: "var",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 22}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 22}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("func(...a)", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "CallExpression",
      callee: {
        type: "Identifier",
        name: "func",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 4}
        }
      },
      arguments: [{
        type: "SpreadElement",
        argument: {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 8},
            end: {line: 1, column: 9}
          }
        },
        loc: {
          start: {line: 1, column: 5},
          end: {line: 1, column: 9}
        }
      }],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 10}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 10}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 10}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("func(a, ...b)", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "CallExpression",
      callee: {
        type: "Identifier",
        name: "func",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 4}
        }
      },
      arguments: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 6}
          }
        },
        {
          type: "SpreadElement",
          argument: {
            type: "Identifier",
            name: "b",
            loc: {
              start: {line: 1, column: 11},
              end: {line: 1, column: 12}
            }
          },
          loc: {
            start: {line: 1, column: 8},
            end: {line: 1, column: 12}
          }
        }
      ],
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  }
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("func(...a, b)", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 13}
  },
  body: [{
    type: "ExpressionStatement",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 13}
    },
    expression: {
      type: "CallExpression",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 13}
      },
      callee: {
        type: "Identifier",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 4}
        },
        name: "func"
      },
      arguments: [
        {
          type: "SpreadElement",
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 9}
          },
          argument: {
            type: "Identifier",
            loc: {
              start: {line: 1, column: 8},
              end: {line: 1, column: 9}
            },
            name: "a"
          }
        },
        {
          type: "Identifier",
          loc: {
            start: {line: 1, column: 11},
            end: {line: 1, column: 12}
          },
          name: "b"
        }
      ]
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

test("/[a-z]/u", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        regex: {
          pattern: "[a-z]",
          flags: "u"
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 8
          }
        }
      }
    }
  ]
}, {
  locations: true,
  ecmaVersion: 6
});


test("do {} while (false) foo();", {
  type: "Program",
  start: 0,
  end: 26,
  body: [
    {
      type: "DoWhileStatement",
      start: 0,
      end: 19,
      body: {
        type: "BlockStatement",
        start: 3,
        end: 5,
        body: []
      },
      test: {
        type: "Literal",
        start: 13,
        end: 18,
        value: false,
        raw: "false"
      }
    },
    {
      type: "ExpressionStatement",
      start: 20,
      end: 26,
      expression: {
        type: "CallExpression",
        start: 20,
        end: 25,
        callee: {
          type: "Identifier",
          start: 20,
          end: 23,
          name: "foo"
        },
        arguments: []
      }
    }
  ]
}, {
  ecmaVersion: 6
});

// Harmony Invalid syntax

testFail("0o", "Expected number in radix 8 (1:2)", {ecmaVersion: 6});

testFail("0o1a", "Identifier directly after number (1:3)", {ecmaVersion: 6});

testFail("0o9", "Expected number in radix 8 (1:2)", {ecmaVersion: 6});

testFail("0o18", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("0O", "Expected number in radix 8 (1:2)", {ecmaVersion: 6});

testFail("0O1a", "Identifier directly after number (1:3)", {ecmaVersion: 6});

testFail("0O9", "Expected number in radix 8 (1:2)", {ecmaVersion: 6});

testFail("0O18", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("0b", "Expected number in radix 2 (1:2)", {ecmaVersion: 6});

testFail("0b1a", "Identifier directly after number (1:3)", {ecmaVersion: 6});

testFail("0b9", "Expected number in radix 2 (1:2)", {ecmaVersion: 6});

testFail("0b18", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("0b12", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("0B", "Expected number in radix 2 (1:2)", {ecmaVersion: 6});

testFail("0B1a", "Identifier directly after number (1:3)", {ecmaVersion: 6});

testFail("0B9", "Expected number in radix 2 (1:2)", {ecmaVersion: 6});

testFail("0B18", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("0B12", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("\"\\u{110000}\"", "Unexpected token (1:0)", {ecmaVersion: 6});

testFail("\"\\u{}\"", "Bad character escape sequence (1:0)", {ecmaVersion: 6});

testFail("\"\\u{FFFF\"", "Bad character escape sequence (1:0)", {ecmaVersion: 6});

testFail("\"\\u{FFZ}\"", "Bad character escape sequence (1:0)", {ecmaVersion: 6});

testFail("[v] += ary", "Assigning to rvalue (1:0)", {ecmaVersion: 6});

testFail("[2] = 42", "Assigning to rvalue (1:1)", {ecmaVersion: 6});

testFail("({ obj:20 }) = 42", "Assigning to rvalue (1:7)", {ecmaVersion: 6});

testFail("( { get x() {} } ) = 0", "Unexpected token (1:8)", {ecmaVersion: 6});

testFail("x \n is y", "Unexpected token (2:4)", {ecmaVersion: 6});

testFail("x \n isnt y", "Unexpected token (2:6)", {ecmaVersion: 6});

testFail("function default() {}", "Unexpected token (1:9)", {ecmaVersion: 6});

testFail("function hello() {'use strict'; ({ i: 10, s(eval) { } }); }", "Defining 'eval' in strict mode (1:44)", {ecmaVersion: 6});

testFail("function a() { \"use strict\"; ({ b(t, t) { } }); }", "Argument name clash in strict mode (1:37)", {ecmaVersion: 6});

testFail("var super", "The keyword 'super' is reserved (1:4)", {ecmaVersion: 6, forbidReserved: true});

testFail("var default", "Unexpected token (1:4)", {ecmaVersion: 6});

testFail("let default", "Unexpected token (1:4)", {ecmaVersion: 6});

testFail("const default", "Unexpected token (1:6)", {ecmaVersion: 6});

testFail("\"use strict\"; ({ v: eval }) = obj", "Assigning to eval in strict mode (1:20)", {ecmaVersion: 6});

testFail("\"use strict\"; ({ v: arguments }) = obj", "Assigning to arguments in strict mode (1:20)", {ecmaVersion: 6});

testFail("for (let x = 42 in list) process(x);", "Unexpected token (1:16)", {ecmaVersion: 6});

testFail("for (let x = 42 of list) process(x);", "Unexpected token (1:16)", {ecmaVersion: 6});

testFail("import foo", "Unexpected token (1:10)", {ecmaVersion: 6});

testFail("import { foo, bar }", "Unexpected token (1:19)", {ecmaVersion: 6});

testFail("import foo from bar", "Unexpected token (1:16)", {ecmaVersion: 6});

testFail("((a)) => 42", "Unexpected token (1:6)", {ecmaVersion: 6});

testFail("(a, (b)) => 42", "Unexpected token (1:9)", {ecmaVersion: 6});

testFail("\"use strict\"; (eval = 10) => 42", "Assigning to eval in strict mode (1:15)", {ecmaVersion: 6});

testFail("\"use strict\"; eval => 42", "Defining 'eval' in strict mode (1:14)", {ecmaVersion: 6});

testFail("\"use strict\"; arguments => 42", "Defining 'arguments' in strict mode (1:14)", {ecmaVersion: 6});

testFail("\"use strict\"; (eval, a) => 42", "Defining 'eval' in strict mode (1:15)", {ecmaVersion: 6});

testFail("\"use strict\"; (arguments, a) => 42", "Defining 'arguments' in strict mode (1:15)", {ecmaVersion: 6});

testFail("\"use strict\"; (eval, a = 10) => 42", "Defining 'eval' in strict mode (1:15)", {ecmaVersion: 6});

testFail("\"use strict\"; (a, a) => 42", "Argument name clash in strict mode (1:18)", {ecmaVersion: 6});

testFail("\"use strict\"; (a) => 00", "Invalid number (1:21)", {ecmaVersion: 6});

testFail("() <= 42", "Unexpected token (1:1)", {ecmaVersion: 6});

testFail("(10) => 00", "Unexpected token (1:1)", {ecmaVersion: 6});

testFail("(10, 20) => 00", "Unexpected token (1:1)", {ecmaVersion: 6});

testFail("yield v", "Unexpected token (1:6)", {ecmaVersion: 6});

testFail("yield 10", "Unexpected token (1:6)", {ecmaVersion: 6});

test("yield* 10", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "BinaryExpression",
      operator: "*",
      left: {
        type: "Identifier",
        name: "yield",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 5}
        }
      },
      right: {
        type: "Literal",
        value: 10,
        raw: "10",
        loc: {
          start: {line: 1, column: 7},
          end: {line: 1, column: 9}
        }
      },
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 9}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 9}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 9}
  }
}, {
  ecmaVersion: 6,
  loose: false,
  ranges: true,
  locations: true
});

test("e => yield* 10", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression",
      id: null,
      params: [{
        type: "Identifier",
        name: "e",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 1}
        }
      }],
      defaults: [],
      body: {
        type: "BinaryExpression",
        operator: "*",
        left: {
          type: "Identifier",
          name: "yield",
          loc: {
            start: {line: 1, column: 5},
            end: {line: 1, column: 10}
          }
        },
        right: {
          type: "Literal",
          value: 10,
          raw: "10",
          loc: {
            start: {line: 1, column: 12},
            end: {line: 1, column: 14}
          }
        },
        loc: {
          start: {line: 1, column: 5},
          end: {line: 1, column: 14}
        }
      },
      rest: null,
      generator: false,
      expression: true,
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 14}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 14}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 14}
  }
}, {
  ecmaVersion: 6,
  loose: false,
  ranges: true,
  locations: true
});

testFail("(function () { yield 10 })", "Unexpected token (1:21)", {ecmaVersion: 6});

test("(function () { yield* 10 })", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "FunctionExpression",
      id: null,
      params: [],
      defaults: [],
      body: {
        type: "BlockStatement",
        body: [{
          type: "ExpressionStatement",
          expression: {
            type: "BinaryExpression",
            operator: "*",
            left: {
              type: "Identifier",
              name: "yield",
              loc: {
                start: {line: 1, column: 15},
                end: {line: 1, column: 20}
              }
            },
            right: {
              type: "Literal",
              value: 10,
              raw: "10",
              loc: {
                start: {line: 1, column: 22},
                end: {line: 1, column: 24}
              }
            },
            loc: {
              start: {line: 1, column: 15},
              end: {line: 1, column: 24}
            }
          },
          loc: {
            start: {line: 1, column: 15},
            end: {line: 1, column: 24}
          }
        }],
        loc: {
          start: {line: 1, column: 13},
          end: {line: 1, column: 26}
        }
      },
      rest: null,
      generator: false,
      expression: false,
      loc: {
        start: {line: 1, column: 1},
        end: {line: 1, column: 26}
      }
    },
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 27}
    }
  }],
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 27}
  }
}, {
  ecmaVersion: 6,
  loose: false,
  ranges: true,
  locations: true
});

testFail("(function() { \"use strict\"; f(yield v) })", "Unexpected token (1:36)", {ecmaVersion: 6});

testFail("var obj = { *test** }", "Unexpected token (1:17)", {ecmaVersion: 6});

testFail("class A extends yield B { }", "Unexpected token (1:22)", {ecmaVersion: 6});

testFail("class default", "Unexpected token (1:6)", {ecmaVersion: 6});

testFail("`test", "Unterminated template (1:0)", {ecmaVersion: 6});

testFail("switch `test`", "Unexpected token (1:7)", {ecmaVersion: 6});

testFail("`hello ${10 `test`", "Unexpected token (1:18)", {ecmaVersion: 6});

testFail("`hello ${10;test`", "Unexpected token (1:11)", {ecmaVersion: 6});

testFail("function a() 1 // expression closure is not supported", "Unexpected token (1:13)", {ecmaVersion: 6});

testFail("[for (let x of []) x]", "Unexpected token (1:6)", {ecmaVersion: 7});

testFail("[for (const x of []) x]", "Unexpected token (1:6)", {ecmaVersion: 7});

testFail("[for (var x of []) x]", "Unexpected token (1:6)", {ecmaVersion: 7});

testFail("[for (a in []) x] // (a,b) ", "Unexpected token (1:8)", {ecmaVersion: 7});

testFail("var a = [if (x) x]", "Unexpected token (1:9)", {ecmaVersion: 6});

testFail("[for (x of [])]  // no expression", "Unexpected token (1:14)", {ecmaVersion: 7});

testFail("({ \"chance\" }) = obj", "Unexpected token (1:12)", {ecmaVersion: 6});

testFail("({ 42 }) = obj", "Unexpected token (1:6)", {ecmaVersion: 6});

testFail("function f(a, ...b, c)", "Unexpected token (1:18)", {ecmaVersion: 6});

testFail("function f(a, ...b = 0)", "Unexpected token (1:19)", {ecmaVersion: 6});

testFail("function x(...{ a }){}", "Unexpected token (1:14)", {ecmaVersion: 6});

testFail("\"use strict\"; function x(a, { a }){}", "Argument name clash in strict mode (1:30)", {ecmaVersion: 6});

testFail("\"use strict\"; function x({ b: { a } }, [{ b: { a } }]){}", "Argument name clash in strict mode (1:47)", {ecmaVersion: 6});

testFail("\"use strict\"; function x(a, ...[a]){}", "Argument name clash in strict mode (1:32)", {ecmaVersion: 6});

testFail("(...a, b) => {}", "Unexpected token (1:1)", {ecmaVersion: 6});

testFail("([ 5 ]) => {}", "Unexpected token (1:3)", {ecmaVersion: 6});

testFail("({ 5 }) => {}", "Unexpected token (1:5)", {ecmaVersion: 6});

testFail("(...[ 5 ]) => {}", "Unexpected token (1:6)", {ecmaVersion: 6});

testFail("[...{ a }] = b", "Unexpected token (1:4)", {ecmaVersion: 6});

testFail("[...a, b] = c", "Unexpected token (1:1)", {ecmaVersion: 6});

testFail("({ t(eval) { \"use strict\"; } });", "Defining 'eval' in strict mode (1:5)", {ecmaVersion: 6});

testFail("\"use strict\"; `${test}\\02`;", "Octal literal in strict mode (1:22)", {ecmaVersion: 6});

testFail("if (1) import \"acorn\";", "'import' and 'export' may only appear at the top level (1:7)", {ecmaVersion: 6});

test("[...a, ] = b", {
  type: "Program",
  loc: {
    start: {line: 1, column: 0},
    end: {line: 1, column: 12}
  },
  body: [{
    type: "ExpressionStatement",
    loc: {
      start: {line: 1, column: 0},
      end: {line: 1, column: 12}
    },
    expression: {
      type: "AssignmentExpression",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 12}
      },
      operator: "=",
      left: {
        type: "ArrayPattern",
        loc: {
          start: {line: 1, column: 0},
          end: {line: 1, column: 8}
        },
        elements: [{
          type: "SpreadElement",
          loc: {
            start: {line: 1, column: 1},
            end: {line: 1, column: 5}
          },
          argument: {
            type: "Identifier",
            loc: {
              start: {line: 1, column: 4},
              end: {line: 1, column: 5}
            },
            name: "a"
          }
        }]
      },
      right: {
        type: "Identifier",
        loc: {
          start: {line: 1, column: 11},
          end: {line: 1, column: 12}
        },
        name: "b"
      }
    }
  }]
}, {
  ecmaVersion: 6,
  ranges: true,
  locations: true
});

testFail("if (b,...a, );", "Unexpected token (1:12)", {ecmaVersion: 6});

testFail("(b, ...a)", "Unexpected token (1:9)", {ecmaVersion: 6});

testFail("switch (cond) { case 10: let a = 20; ", "Unexpected token (1:37)", {ecmaVersion: 6});

testFail("\"use strict\"; (eval) => 42", "Defining 'eval' in strict mode (1:15)", {ecmaVersion: 6});

testFail("(eval) => { \"use strict\"; 42 }", "Defining 'eval' in strict mode (1:1)", {ecmaVersion: 6});

testFail("({ get test() { } }) => 42", "Unexpected token (1:7)", {ecmaVersion: 6});

/* Regression tests */

// # https://github.com/marijnh/acorn/issues/127
test('doSmth(`${x} + ${y} = ${x + y}`)', {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "CallExpression",
      callee: {
        type: "Identifier",
        name: "doSmth"
      },
      arguments: [{
        type: "TemplateLiteral",
        expressions: [
          {
            type: "Identifier",
            name: "x"
          },
          {
            type: "Identifier",
            name: "y"
          },
          {
            type: "BinaryExpression",
            left: {
              type: "Identifier",
              name: "x"
            },
            operator: "+",
            right: {
              type: "Identifier",
              name: "y"
            }
          }
        ],
        quasis: [
          {
            type: "TemplateElement",
            value: {cooked: "", raw: ""},
            tail: false
          },
          {
            type: "TemplateElement",
            value: {cooked: " + ", raw: " + "},
            tail: false
          },
          {
            type: "TemplateElement",
            value: {cooked: " = ", raw: " = "},
            tail: false
          },
          {
            type: "TemplateElement",
            value: {cooked: "", raw: ""},
            tail: true
          }
        ]
      }]
    }
  }]
}, {ecmaVersion: 6});

// # https://github.com/marijnh/acorn/issues/129
test('function normal(x, y = 10) {}', {
  type: "Program",
  body: [{
    type: "FunctionDeclaration",
    id: {
      type: "Identifier",
      name: "normal"
    },
    params: [
      {
        type: "Identifier",
        name: "x"
      },
      {
        type: "Identifier",
        name: "y"
      }
    ],
    defaults: [
      null,
      {
        type: "Literal",
        value: 10,
        raw: "10"
      }
    ],
    rest: null,
    generator: false,
    body: {
      type: "BlockStatement",
      body: []
    },
    expression: false
  }]
}, {ecmaVersion: 6});

test("'use strict'; function f([x,,z]) {}", {}, {ecmaVersion: 6});

// test preserveParens option with arrow functions
test("() => 42", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ArrowFunctionExpression"
    }
  }]
}, {ecmaVersion: 6, preserveParens: true});

// test preserveParens with generators
test("(for (x of array) for (y of array2) if (x === test) x)", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "ComprehensionExpression"
    }
  }]
}, {ecmaVersion: 7, preserveParens: true});

// https://github.com/marijnh/acorn/issues/161
test("import foo, * as bar from 'baz';", {
  type: "Program",
  body: [{
    type: "ImportDeclaration",
    specifiers: [
      {
        type: "ImportSpecifier",
        id: {
          type: "Identifier",
          name: "foo"
        },
        name: null,
        default: true
      },
      {
        type: "ImportBatchSpecifier",
        name: {
          type: "Identifier",
          name: "bar"
        }
      }
    ],
    source: {
      type: "Literal",
      value: "baz",
      raw: "'baz'"
    }
  }]
}, {ecmaVersion: 6});

// https://github.com/marijnh/acorn/issues/173
test("`{${x}}`, `}`", {
  type: "Program",
  body: [{
    type: "ExpressionStatement",
    expression: {
      type: "SequenceExpression",
      expressions: [
        {
          type: "TemplateLiteral",
          expressions: [{
            type: "Identifier",
            name: "x"
          }],
          quasis: [
            {
              type: "TemplateElement",
              value: {cooked: "{", raw: "{"},
              tail: false
            },
            {
              type: "TemplateElement",
              value: {cooked: "}", raw: "}"},
              tail: true
            }
          ]
        },
        {
          type: "TemplateLiteral",
          expressions: [],
          quasis: [{
            type: "TemplateElement",
            value: {cooked: "}", raw: "}"},
            tail: true
          }]
        }
      ]
    }
  }]
}, {ecmaVersion: 6});
