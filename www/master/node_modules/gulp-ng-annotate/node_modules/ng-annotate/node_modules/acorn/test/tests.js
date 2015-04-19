// Tests largely based on those of Esprima
// (http://esprima.org/test/)

if (typeof exports != "undefined") {
  var driver = require("./driver.js");
  var test = driver.test, testFail = driver.testFail, testAssert = driver.testAssert, misMatch = driver.misMatch;
  var acorn = require("..");
}

test("this\n", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "ThisExpression",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 0
    }
  }
});

test("null\n", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: null,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 0
    }
  }
});

test("\n    42\n\n", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 4
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 4
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 4,
      column: 0
    }
  }
});

test("/foobar/", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: /foobar/,
        regex: {
          pattern: "foobar",
          flags: ""
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
});

test("/[a-z]/g", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: /[a-z]/,
        regex: {
          pattern: "[a-z]",
          flags: "g"
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
});

test("(1 + 2 ) * 3", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Literal",
            value: 1,
            loc: {
              start: {
                line: 1,
                column: 1
              },
              end: {
                line: 1,
                column: 2
              }
            }
          },
          operator: "+",
          right: {
            type: "Literal",
            value: 2,
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 1
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        operator: "*",
        right: {
          type: "Literal",
          value: 3,
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 12
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 12
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 12
    }
  }
});

test("(1 + 2 ) * 3", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "ParenthesizedExpression",
          expression: {
            type: "BinaryExpression",
            left: {
              type: "Literal",
              value: 1,
              loc: {
                start: {
                  line: 1,
                  column: 1
                },
                end: {
                  line: 1,
                  column: 2
                }
              }
            },
            operator: "+",
            right: {
              type: "Literal",
              value: 2,
              loc: {
                start: {
                  line: 1,
                  column: 5
                },
                end: {
                  line: 1,
                  column: 6
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 1
              },
              end: {
                line: 1,
                column: 6
              }
            }
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
        },
        operator: "*",
        right: {
          type: "Literal",
          value: 3,
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 12
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 12
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 12
    }
  }
}, {
  locations: true,
  preserveParens: true
});

testFail("(x) = 23", "Assigning to rvalue (1:0)", { preserveParens: true });

test("x = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x = [ ]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x = [ 42 ]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [
            {
              type: "Literal",
              value: 42,
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 8
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 10
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 10
    }
  }
});

test("x = [ 42, ]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [
            {
              type: "Literal",
              value: 42,
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 8
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("x = [ ,, 42 ]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [
            null,
            null,
            {
              type: "Literal",
              value: 42,
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 11
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("x = [ 1, 2, 3, ]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [
            {
              type: "Literal",
              value: 1,
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 7
                }
              }
            },
            {
              type: "Literal",
              value: 2,
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            {
              type: "Literal",
              value: 3,
              loc: {
                start: {
                  line: 1,
                  column: 12
                },
                end: {
                  line: 1,
                  column: 13
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 16
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("x = [ 1, 2,, 3, ]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [
            {
              type: "Literal",
              value: 1,
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 7
                }
              }
            },
            {
              type: "Literal",
              value: 2,
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            null,
            {
              type: "Literal",
              value: 3,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 14
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
});

test("日本語 = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "日本語",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 3
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 8
            }
          }
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
  ],
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
});

test("T‿ = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "T‿",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("T‌ = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "T‌",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("T‍ = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "T‍",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("ⅣⅡ = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "ⅣⅡ",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("ⅣⅡ = []", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "ⅣⅡ",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        right: {
          type: "ArrayExpression",
          elements: [],
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x = {}", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x = { }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x = { answer: 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "answer",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 14
                  },
                  end: {
                    line: 1,
                    column: 16
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 18
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 18
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 18
    }
  }
});

test("x = { if: 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "if",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 8
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("x = { true: 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "true",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 10
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 16
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("x = { false: 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "false",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 11
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 13
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
});

test("x = { null: 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "null",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 10
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 16
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("x = { \"answer\": 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Literal",
                value: "answer",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 16
                  },
                  end: {
                    line: 1,
                    column: 18
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 20
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 20
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 20
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 20
    }
  }
});

test("x = { x: 1, x: 2 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 7
                  }
                }
              },
              value: {
                type: "Literal",
                value: 1,
                loc: {
                  start: {
                    line: 1,
                    column: 9
                  },
                  end: {
                    line: 1,
                    column: 10
                  }
                }
              },
              kind: "init"
            },
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 13
                  }
                }
              },
              value: {
                type: "Literal",
                value: 2,
                loc: {
                  start: {
                    line: 1,
                    column: 15
                  },
                  end: {
                    line: 1,
                    column: 16
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 18
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 18
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 18
    }
  }
});

test("x = { get width() { return m_width } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "width",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ReturnStatement",
                      argument: {
                        type: "Identifier",
                        name: "m_width",
                        loc: {
                          start: {
                            line: 1,
                            column: 27
                          },
                          end: {
                            line: 1,
                            column: 34
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 20
                        },
                        end: {
                          line: 1,
                          column: 34
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 18
                    },
                    end: {
                      line: 1,
                      column: 36
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 15
                  },
                  end: {
                    line: 1,
                    column: 36
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 38
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 38
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 38
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 38
    }
  }
});

test("x = { get undef() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "undef",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 18
                    },
                    end: {
                      line: 1,
                      column: 20
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 15
                  },
                  end: {
                    line: 1,
                    column: 20
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 22
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("x = { get if() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "if",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 15
                    },
                    end: {
                      line: 1,
                      column: 17
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 17
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 19
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 19
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 19
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 19
    }
  }
});

test("x = { get true() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "true",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 17
                    },
                    end: {
                      line: 1,
                      column: 19
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 14
                  },
                  end: {
                    line: 1,
                    column: 19
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 21
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 21
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 21
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("x = { get false() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "false",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 18
                    },
                    end: {
                      line: 1,
                      column: 20
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 15
                  },
                  end: {
                    line: 1,
                    column: 20
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 22
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("x = { get null() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "null",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 17
                    },
                    end: {
                      line: 1,
                      column: 19
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 14
                  },
                  end: {
                    line: 1,
                    column: 19
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 21
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 21
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 21
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("x = { get \"undef\"() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Literal",
                value: "undef",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 17
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 20
                    },
                    end: {
                      line: 1,
                      column: 22
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 17
                  },
                  end: {
                    line: 1,
                    column: 22
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 24
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
});

test("x = { get 10() {} }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Literal",
                value: 10,
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              kind: "get",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [],
                body: {
                  type: "BlockStatement",
                  body: [],
                  loc: {
                    start: {
                      line: 1,
                      column: 15
                    },
                    end: {
                      line: 1,
                      column: 17
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 17
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 19
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 19
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 19
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 19
    }
  }
});

test("x = { set width(w) { m_width = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "width",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 16
                      },
                      end: {
                        line: 1,
                        column: 17
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_width",
                          loc: {
                            start: {
                              line: 1,
                              column: 21
                            },
                            end: {
                              line: 1,
                              column: 28
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 31
                            },
                            end: {
                              line: 1,
                              column: 32
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 21
                          },
                          end: {
                            line: 1,
                            column: 32
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 21
                        },
                        end: {
                          line: 1,
                          column: 32
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 19
                    },
                    end: {
                      line: 1,
                      column: 34
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 15
                  },
                  end: {
                    line: 1,
                    column: 34
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 36
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 36
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 36
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 36
    }
  }
});

test("x = { set if(w) { m_if = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "if",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 13
                      },
                      end: {
                        line: 1,
                        column: 14
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_if",
                          loc: {
                            start: {
                              line: 1,
                              column: 18
                            },
                            end: {
                              line: 1,
                              column: 22
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 25
                            },
                            end: {
                              line: 1,
                              column: 26
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 18
                          },
                          end: {
                            line: 1,
                            column: 26
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 18
                        },
                        end: {
                          line: 1,
                          column: 26
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 16
                    },
                    end: {
                      line: 1,
                      column: 28
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 28
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 30
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 30
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 30
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 30
    }
  }
});

test("x = { set true(w) { m_true = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "true",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 15
                      },
                      end: {
                        line: 1,
                        column: 16
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_true",
                          loc: {
                            start: {
                              line: 1,
                              column: 20
                            },
                            end: {
                              line: 1,
                              column: 26
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 29
                            },
                            end: {
                              line: 1,
                              column: 30
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 20
                          },
                          end: {
                            line: 1,
                            column: 30
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 20
                        },
                        end: {
                          line: 1,
                          column: 30
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 18
                    },
                    end: {
                      line: 1,
                      column: 32
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 14
                  },
                  end: {
                    line: 1,
                    column: 32
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 34
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 34
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 34
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 34
    }
  }
});

test("x = { set false(w) { m_false = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "false",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 16
                      },
                      end: {
                        line: 1,
                        column: 17
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_false",
                          loc: {
                            start: {
                              line: 1,
                              column: 21
                            },
                            end: {
                              line: 1,
                              column: 28
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 31
                            },
                            end: {
                              line: 1,
                              column: 32
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 21
                          },
                          end: {
                            line: 1,
                            column: 32
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 21
                        },
                        end: {
                          line: 1,
                          column: 32
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 19
                    },
                    end: {
                      line: 1,
                      column: 34
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 15
                  },
                  end: {
                    line: 1,
                    column: 34
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 36
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 36
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 36
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 36
    }
  }
});

test("x = { set null(w) { m_null = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "null",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 15
                      },
                      end: {
                        line: 1,
                        column: 16
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_null",
                          loc: {
                            start: {
                              line: 1,
                              column: 20
                            },
                            end: {
                              line: 1,
                              column: 26
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 29
                            },
                            end: {
                              line: 1,
                              column: 30
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 20
                          },
                          end: {
                            line: 1,
                            column: 30
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 20
                        },
                        end: {
                          line: 1,
                          column: 30
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 18
                    },
                    end: {
                      line: 1,
                      column: 32
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 14
                  },
                  end: {
                    line: 1,
                    column: 32
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 34
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 34
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 34
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 34
    }
  }
});

test("x = { set \"null\"(w) { m_null = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Literal",
                value: "null",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 16
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 17
                      },
                      end: {
                        line: 1,
                        column: 18
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_null",
                          loc: {
                            start: {
                              line: 1,
                              column: 22
                            },
                            end: {
                              line: 1,
                              column: 28
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 31
                            },
                            end: {
                              line: 1,
                              column: 32
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 22
                          },
                          end: {
                            line: 1,
                            column: 32
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 22
                        },
                        end: {
                          line: 1,
                          column: 32
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 20
                    },
                    end: {
                      line: 1,
                      column: 34
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 16
                  },
                  end: {
                    line: 1,
                    column: 34
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 36
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 36
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 36
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 36
    }
  }
});

test("x = { set 10(w) { m_null = w } }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Literal",
                value: 10,
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              kind: "set",
              value: {
                type: "FunctionExpression",
                id: null,
                params: [
                  {
                    type: "Identifier",
                    name: "w",
                    loc: {
                      start: {
                        line: 1,
                        column: 13
                      },
                      end: {
                        line: 1,
                        column: 14
                      }
                    }
                  }
                ],
                body: {
                  type: "BlockStatement",
                  body: [
                    {
                      type: "ExpressionStatement",
                      expression: {
                        type: "AssignmentExpression",
                        operator: "=",
                        left: {
                          type: "Identifier",
                          name: "m_null",
                          loc: {
                            start: {
                              line: 1,
                              column: 18
                            },
                            end: {
                              line: 1,
                              column: 24
                            }
                          }
                        },
                        right: {
                          type: "Identifier",
                          name: "w",
                          loc: {
                            start: {
                              line: 1,
                              column: 27
                            },
                            end: {
                              line: 1,
                              column: 28
                            }
                          }
                        },
                        loc: {
                          start: {
                            line: 1,
                            column: 18
                          },
                          end: {
                            line: 1,
                            column: 28
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 18
                        },
                        end: {
                          line: 1,
                          column: 28
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 16
                    },
                    end: {
                      line: 1,
                      column: 30
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 12
                  },
                  end: {
                    line: 1,
                    column: 30
                  }
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 32
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 32
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 32
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 32
    }
  }
});

test("x = { get: 42 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "get",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 9
                  }
                }
              },
              value: {
                type: "Literal",
                value: 42,
                loc: {
                  start: {
                    line: 1,
                    column: 11
                  },
                  end: {
                    line: 1,
                    column: 13
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 15
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 15
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 15
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 15
    }
  }
});

test("x = { set: 43 }", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "ObjectExpression",
          properties: [
            {
              type: "Property",
              key: {
                type: "Identifier",
                name: "set",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 9
                  }
                }
              },
              value: {
                type: "Literal",
                value: 43,
                loc: {
                  start: {
                    line: 1,
                    column: 11
                  },
                  end: {
                    line: 1,
                    column: 13
                  }
                }
              },
              kind: "init"
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 15
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 15
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 15
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 15
    }
  }
});

test("/* block comment */ 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 1,
            column: 20
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 20
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("42 /*The*/ /*Answer*/", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("42 /*the*/ /*answer*/", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("/* multiline\ncomment\nshould\nbe\nignored */ 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 5,
            column: 11
          },
          end: {
            line: 5,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 5,
          column: 11
        },
        end: {
          line: 5,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 5,
      column: 13
    }
  }
});

test("/*a\r\nb*/ 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 4
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 4
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 6
    }
  }
});

test("/*a\rb*/ 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 4
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 4
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 6
    }
  }
});

test("/*a\nb*/ 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 4
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 4
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 6
    }
  }
});

test("/*a\nc*/ 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 4
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 4
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 6
    }
  }
});

test("// line comment\n42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 0
          },
          end: {
            line: 2,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 0
        },
        end: {
          line: 2,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 2
    }
  }
});

test("42 // line comment", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 18
    }
  }
});

test("// Hello, world!\n42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 0
          },
          end: {
            line: 2,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 0
        },
        end: {
          line: 2,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 2
    }
  }
});

test("// Hello, world!\n", {
  type: "Program",
  body: [],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 0
    }
  }
});

test("// Hallo, world!\n", {
  type: "Program",
  body: [],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 0
    }
  }
});

test("//\n42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 2,
            column: 0
          },
          end: {
            line: 2,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 2,
          column: 0
        },
        end: {
          line: 2,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 2
    }
  }
});

test("//", {
  type: "Program",
  body: [],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("// ", {
  type: "Program",
  body: [],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("/**/42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 4
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("// Hello, world!\n\n//   Another hello\n42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 4,
            column: 0
          },
          end: {
            line: 4,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 4,
          column: 0
        },
        end: {
          line: 4,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 4,
      column: 2
    }
  }
});

test("if (x) { // Some comment\ndoThat(); }", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      consequent: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "doThat",
                loc: {
                  start: {
                    line: 2,
                    column: 0
                  },
                  end: {
                    line: 2,
                    column: 6
                  }
                }
              },
              arguments: [],
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 8
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 9
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 2,
            column: 11
          }
        }
      },
      alternate: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 11
    }
  }
});

test("switch (answer) { case 42: /* perfect */ bingo() }", {
  type: "Program",
  body: [
    {
      type: "SwitchStatement",
      discriminant: {
        type: "Identifier",
        name: "answer",
        loc: {
          start: {
            line: 1,
            column: 8
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      cases: [
        {
          type: "SwitchCase",
          consequent: [
            {
              type: "ExpressionStatement",
              expression: {
                type: "CallExpression",
                callee: {
                  type: "Identifier",
                  name: "bingo",
                  loc: {
                    start: {
                      line: 1,
                      column: 41
                    },
                    end: {
                      line: 1,
                      column: 46
                    }
                  }
                },
                arguments: [],
                loc: {
                  start: {
                    line: 1,
                    column: 41
                  },
                  end: {
                    line: 1,
                    column: 48
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 41
                },
                end: {
                  line: 1,
                  column: 48
                }
              }
            }
          ],
          test: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 25
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 48
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 50
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 50
    }
  }
});

test("0", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 0,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 1
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 1
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 1
    }
  }
});

test("3", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 3,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 1
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 1
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 1
    }
  }
});

test("5", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 5,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 1
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 1
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 1
    }
  }
});

test("42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 42,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test(".14", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 0.14,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("3.14159", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 3.14159,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("6.02214179e+23", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 6.02214179e+23,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("1.492417830e-10", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 1.49241783e-10,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 15
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 15
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 15
    }
  }
});

test("0x0", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 0,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("0e+100", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 0,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("0xabc", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 2748,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("0xdef", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 3567,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("0X1A", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 26,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 4
    }
  }
});

test("0x10", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 16,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 4
    }
  }
});

test("0x100", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 256,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("0X04", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 4,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 4
    }
  }
});

test("02", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 2,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("012", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 10,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("0012", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: 10,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 4
    }
  }
});

test("\"Hello\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("\"\\n\\r\\t\\v\\b\\f\\\\\\'\\\"\\0\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "\n\r\t\u000b\b\f\\'\"\u0000",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("\"\\u0061\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "a",
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
  ],
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
});

test("\"\\x61\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "a",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("\"Hello\\nworld\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello\nworld",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("\"Hello\\\nworld\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Helloworld",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 6
    }
  }
});

test("\"Hello\\02World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello\u0002World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 15
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 15
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 15
    }
  }
});

test("\"Hello\\012World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello\nWorld",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("\"Hello\\122World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "HelloRWorld",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("\"Hello\\0122World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello\n2World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
});

test("\"Hello\\312World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "HelloÊWorld",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("\"Hello\\412World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello!2World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("\"Hello\\812World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello812World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("\"Hello\\712World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello92World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("\"Hello\\0World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello\u0000World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("\"Hello\\\r\nworld\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Helloworld",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 2,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 6
    }
  }
});

test("\"Hello\\1World\"", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "Hello\u0001World",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("var x = /[a-z]/i", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 16
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 16
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("var x = /[x-z]/i", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 16
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 16
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("var x = /[a-c]/i", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 16
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 16
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("var x = /[P QR]/i", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 17
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 17
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
});

test("var x = /foo\\/bar/", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 18
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 18
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 18
    }
  }
});

test("var x = /=([^=\\s])+/g", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 21
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 21
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 21
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("var x = /[P QR]/\\u0067", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: {},
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 22
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 22
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("new Button", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "NewExpression",
        callee: {
          type: "Identifier",
          name: "Button",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 10
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 10
    }
  }
});

test("new Button()", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "NewExpression",
        callee: {
          type: "Identifier",
          name: "Button",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 12
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 12
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 12
    }
  }
});

test("new new foo", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "NewExpression",
        callee: {
          type: "NewExpression",
          callee: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("new new foo()", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "NewExpression",
        callee: {
          type: "NewExpression",
          callee: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("new foo().bar()", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "NewExpression",
            callee: {
              type: "Identifier",
              name: "foo",
              loc: {
                start: {
                  line: 1,
                  column: 4
                },
                end: {
                  line: 1,
                  column: 7
                }
              }
            },
            arguments: [],
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          property: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 10
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          computed: false,
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 15
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 15
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 15
    }
  }
});

test("new foo[bar]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "NewExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          property: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          computed: true,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 12
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 12
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 12
    }
  }
});

test("new foo.bar()", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "NewExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          property: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          computed: false,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("( new foo).bar()", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "NewExpression",
            callee: {
              type: "Identifier",
              name: "foo",
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 9
                }
              }
            },
            arguments: [],
            loc: {
              start: {
                line: 1,
                column: 2
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          property: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 11
              },
              end: {
                line: 1,
                column: 14
              }
            }
          },
          computed: false,
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 16
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 16
    }
  }
});

test("foo(bar, baz)", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "Identifier",
          name: "foo",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 3
            }
          }
        },
        arguments: [
          {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          {
            type: "Identifier",
            name: "baz",
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 12
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("(    foo  )()", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "Identifier",
          name: "foo",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 8
            }
          }
        },
        arguments: [],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("universe.milkyway", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "Identifier",
          name: "milkyway",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
});

test("universe.milkyway.solarsystem", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "MemberExpression",
          object: {
            type: "Identifier",
            name: "universe",
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
          },
          property: {
            type: "Identifier",
            name: "milkyway",
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 17
              }
            }
          },
          computed: false,
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        property: {
          type: "Identifier",
          name: "solarsystem",
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 29
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 29
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
});

test("universe.milkyway.solarsystem.Earth", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "MemberExpression",
          object: {
            type: "MemberExpression",
            object: {
              type: "Identifier",
              name: "universe",
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
            },
            property: {
              type: "Identifier",
              name: "milkyway",
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 17
                }
              }
            },
            computed: false,
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 17
              }
            }
          },
          property: {
            type: "Identifier",
            name: "solarsystem",
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 29
              }
            }
          },
          computed: false,
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 29
            }
          }
        },
        property: {
          type: "Identifier",
          name: "Earth",
          loc: {
            start: {
              line: 1,
              column: 30
            },
            end: {
              line: 1,
              column: 35
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 35
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 35
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 35
    }
  }
});

test("universe[galaxyName, otherUselessName]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "SequenceExpression",
          expressions: [
            {
              type: "Identifier",
              name: "galaxyName",
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 19
                }
              }
            },
            {
              type: "Identifier",
              name: "otherUselessName",
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 37
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 37
            }
          }
        },
        computed: true,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 38
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 38
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 38
    }
  }
});

test("universe[galaxyName]", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "Identifier",
          name: "galaxyName",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 19
            }
          }
        },
        computed: true,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 20
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 20
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 20
    }
  }
});

test("universe[42].galaxies", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "MemberExpression",
          object: {
            type: "Identifier",
            name: "universe",
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
          },
          property: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          computed: true,
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        property: {
          type: "Identifier",
          name: "galaxies",
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 21
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 21
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 21
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("universe(42).galaxies", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "universe",
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
          },
          arguments: [
            {
              type: "Literal",
              value: 42,
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 11
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        property: {
          type: "Identifier",
          name: "galaxies",
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 21
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 21
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 21
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("universe(42).galaxies(14, 3, 77).milkyway", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "CallExpression",
          callee: {
            type: "MemberExpression",
            object: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "universe",
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
              },
              arguments: [
                {
                  type: "Literal",
                  value: 42,
                  loc: {
                    start: {
                      line: 1,
                      column: 9
                    },
                    end: {
                      line: 1,
                      column: 11
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 0
                },
                end: {
                  line: 1,
                  column: 12
                }
              }
            },
            property: {
              type: "Identifier",
              name: "galaxies",
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 21
                }
              }
            },
            computed: false,
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 21
              }
            }
          },
          arguments: [
            {
              type: "Literal",
              value: 14,
              loc: {
                start: {
                  line: 1,
                  column: 22
                },
                end: {
                  line: 1,
                  column: 24
                }
              }
            },
            {
              type: "Literal",
              value: 3,
              loc: {
                start: {
                  line: 1,
                  column: 26
                },
                end: {
                  line: 1,
                  column: 27
                }
              }
            },
            {
              type: "Literal",
              value: 77,
              loc: {
                start: {
                  line: 1,
                  column: 29
                },
                end: {
                  line: 1,
                  column: 31
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 32
            }
          }
        },
        property: {
          type: "Identifier",
          name: "milkyway",
          loc: {
            start: {
              line: 1,
              column: 33
            },
            end: {
              line: 1,
              column: 41
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 41
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 41
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 41
    }
  }
});

test("earth.asia.Indonesia.prepareForElection(2014)", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "MemberExpression",
            object: {
              type: "MemberExpression",
              object: {
                type: "Identifier",
                name: "earth",
                loc: {
                  start: {
                    line: 1,
                    column: 0
                  },
                  end: {
                    line: 1,
                    column: 5
                  }
                }
              },
              property: {
                type: "Identifier",
                name: "asia",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 10
                  }
                }
              },
              computed: false,
              loc: {
                start: {
                  line: 1,
                  column: 0
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            property: {
              type: "Identifier",
              name: "Indonesia",
              loc: {
                start: {
                  line: 1,
                  column: 11
                },
                end: {
                  line: 1,
                  column: 20
                }
              }
            },
            computed: false,
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          property: {
            type: "Identifier",
            name: "prepareForElection",
            loc: {
              start: {
                line: 1,
                column: 21
              },
              end: {
                line: 1,
                column: 39
              }
            }
          },
          computed: false,
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 39
            }
          }
        },
        arguments: [
          {
            type: "Literal",
            value: 2014,
            loc: {
              start: {
                line: 1,
                column: 40
              },
              end: {
                line: 1,
                column: 44
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 45
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 45
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 45
    }
  }
});

test("universe.if", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "Identifier",
          name: "if",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("universe.true", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "Identifier",
          name: "true",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("universe.false", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "Identifier",
          name: "false",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("universe.null", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "MemberExpression",
        object: {
          type: "Identifier",
          name: "universe",
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
        },
        property: {
          type: "Identifier",
          name: "null",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        computed: false,
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("x++", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "++",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("x--", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "--",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("eval++", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "++",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 4
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("eval--", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "--",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 4
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("arguments++", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "++",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "arguments",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("arguments--", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "--",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "arguments",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("++x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "++",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 3
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("--x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "--",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 3
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 3
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 3
    }
  }
});

test("++eval", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "++",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("--eval", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "--",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("++arguments", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "++",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "arguments",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("--arguments", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UpdateExpression",
        operator: "--",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "arguments",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("+x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "+",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 1
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("-x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "-",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 1
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("~x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "~",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 1
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("!x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "!",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 1
            },
            end: {
              line: 1,
              column: 2
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 2
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("void x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "void",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("delete x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "delete",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 8
            }
          }
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
  ],
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
});

test("typeof x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "UnaryExpression",
        operator: "typeof",
        prefix: true,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 8
            }
          }
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
  ],
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
});

test("x * y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "*",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x / y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "/",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x % y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "%",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x + y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "+",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x - y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "-",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x << y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "<<",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x >> y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: ">>",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x >>> y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: ">>>",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x < y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "<",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x > y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: ">",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x <= y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "<=",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x >= y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: ">=",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x in y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "in",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x instanceof y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "instanceof",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("x < y < z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "<",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "<",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x == y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "==",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x != y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "!=",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x === y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "===",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x !== y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "!==",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x & y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "&",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x ^ y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "^",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x | y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "|",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("x + y + z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "+",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "+",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x - y + z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "-",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "+",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x + y - z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "+",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "-",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x - y - z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "-",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "-",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x + y * z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "+",
        right: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          operator: "*",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x + y / z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "+",
        right: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          operator: "/",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x - y % z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "-",
        right: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          operator: "%",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x * y * z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "*",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "*",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x * y / z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "*",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "/",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x * y % z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "*",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "%",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x % y * z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "%",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "*",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x << y << z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "<<",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        operator: "<<",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 10
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("x | y | z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "|",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "|",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x & y & z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "&",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "&",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x ^ y ^ z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "^",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "^",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x & y | z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "&",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        operator: "|",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x | y ^ z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "|",
        right: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          operator: "^",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x | y & z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "|",
        right: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          operator: "&",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x || y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "LogicalExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "||",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x && y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "LogicalExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "&&",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("x || y || z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "LogicalExpression",
        left: {
          type: "LogicalExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "||",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        operator: "||",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 10
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("x && y && z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "LogicalExpression",
        left: {
          type: "LogicalExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "&&",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        operator: "&&",
        right: {
          type: "Identifier",
          name: "z",
          loc: {
            start: {
              line: 1,
              column: 10
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("x || y && z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "LogicalExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "||",
        right: {
          type: "LogicalExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          operator: "&&",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 10
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("x || y ^ z", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "LogicalExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        operator: "||",
        right: {
          type: "BinaryExpression",
          left: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          operator: "^",
          right: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 10
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 10
    }
  }
});

test("y ? 1 : 2", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "ConditionalExpression",
        test: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        consequent: {
          type: "Literal",
          value: 1,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        alternate: {
          type: "Literal",
          value: 2,
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x && y ? 1 : 2", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "ConditionalExpression",
        test: {
          type: "LogicalExpression",
          left: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          operator: "&&",
          right: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        consequent: {
          type: "Literal",
          value: 1,
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        alternate: {
          type: "Literal",
          value: 2,
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("x = 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 6
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("eval = 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "eval",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 4
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("arguments = 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "arguments",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 12
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("x *= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "*=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x /= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "/=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x %= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "%=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x += 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "+=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x -= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "-=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x <<= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "<<=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 8
            }
          }
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
  ],
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
});

test("x >>= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: ">>=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 8
            }
          }
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
  ],
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
});

test("x >>>= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: ">>>=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("x &= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "&=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x ^= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "^=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("x |= 42", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "|=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 1
            }
          }
        },
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 5
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("{ foo }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 2
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 5
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("{ doThis(); doThat(); }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ExpressionStatement",
          expression: {
            type: "CallExpression",
            callee: {
              type: "Identifier",
              name: "doThis",
              loc: {
                start: {
                  line: 1,
                  column: 2
                },
                end: {
                  line: 1,
                  column: 8
                }
              }
            },
            arguments: [],
            loc: {
              start: {
                line: 1,
                column: 2
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "CallExpression",
            callee: {
              type: "Identifier",
              name: "doThat",
              loc: {
                start: {
                  line: 1,
                  column: 12
                },
                end: {
                  line: 1,
                  column: 18
                }
              }
            },
            arguments: [],
            loc: {
              start: {
                line: 1,
                column: 12
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 12
            },
            end: {
              line: 1,
              column: 21
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 23
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 23
    }
  }
});

test("{}", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 2
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 2
    }
  }
});

test("var x", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
});

test("var x, y;", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 7
              },
              end: {
                line: 1,
                column: 8
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 8
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("var x = 42", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 10
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 10
    }
  }
});

test("var eval = 42, arguments = 42", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "eval",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 8
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 11
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "arguments",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 27
              },
              end: {
                line: 1,
                column: 29
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 29
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
});

test("var x = 14, y = 3, z = 1977", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: 14,
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 12
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          init: {
            type: "Literal",
            value: 3,
            loc: {
              start: {
                line: 1,
                column: 16
              },
              end: {
                line: 1,
                column: 17
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 12
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 19
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          init: {
            type: "Literal",
            value: 1977,
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 27
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 27
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 27
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 27
    }
  }
});

test("var implements, interface, package", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "implements",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 14
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "interface",
            loc: {
              start: {
                line: 1,
                column: 16
              },
              end: {
                line: 1,
                column: 25
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 16
            },
            end: {
              line: 1,
              column: 25
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "package",
            loc: {
              start: {
                line: 1,
                column: 27
              },
              end: {
                line: 1,
                column: 34
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 27
            },
            end: {
              line: 1,
              column: 34
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 34
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 34
    }
  }
});

test("var private, protected, public, static", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "private",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 11
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "protected",
            loc: {
              start: {
                line: 1,
                column: 13
              },
              end: {
                line: 1,
                column: 22
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 22
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "public",
            loc: {
              start: {
                line: 1,
                column: 24
              },
              end: {
                line: 1,
                column: 30
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 24
            },
            end: {
              line: 1,
              column: 30
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "static",
            loc: {
              start: {
                line: 1,
                column: 32
              },
              end: {
                line: 1,
                column: 38
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 32
            },
            end: {
              line: 1,
              column: 38
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 38
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 38
    }
  }
});

test(";", {
  type: "Program",
  body: [
    {
      type: "EmptyStatement",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 1
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 1
    }
  }
});

test("x", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 1
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 1
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 1
    }
  }
});

test("x, y", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "SequenceExpression",
        expressions: [
          {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 0
              },
              end: {
                line: 1,
                column: 1
              }
            }
          },
          {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 3
              },
              end: {
                line: 1,
                column: 4
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 4
    }
  }
});

test("\\u0061", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Identifier",
        name: "a",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 6
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 6
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 6
    }
  }
});

test("a\\u0061", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Identifier",
        name: "aa",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 7
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 7
    }
  }
});

test("if (morning) goodMorning()", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Identifier",
        name: "morning",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      consequent: {
        type: "ExpressionStatement",
        expression: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "goodMorning",
            loc: {
              start: {
                line: 1,
                column: 13
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 26
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 26
          }
        }
      },
      alternate: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 26
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 26
    }
  }
});

test("if (morning) (function(){})", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Identifier",
        name: "morning",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      consequent: {
        type: "ExpressionStatement",
        expression: {
          type: "FunctionExpression",
          id: null,
          params: [],
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {
                line: 1,
                column: 24
              },
              end: {
                line: 1,
                column: 26
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 14
            },
            end: {
              line: 1,
              column: 26
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 27
          }
        }
      },
      alternate: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 27
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 27
    }
  }
});

test("if (morning) var x = 0;", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Identifier",
        name: "morning",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      consequent: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 17
                },
                end: {
                  line: 1,
                  column: 18
                }
              }
            },
            init: {
              type: "Literal",
              value: 0,
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 22
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 17
              },
              end: {
                line: 1,
                column: 22
              }
            }
          }
        ],
        kind: "var",
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 23
          }
        }
      },
      alternate: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 23
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 23
    }
  }
});

test("if (morning) function a(){}", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Identifier",
        name: "morning",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      consequent: {
        type: "FunctionDeclaration",
        id: {
          type: "Identifier",
          name: "a",
          loc: {
            start: {
              line: 1,
              column: 22
            },
            end: {
              line: 1,
              column: 23
            }
          }
        },
        params: [],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {
              line: 1,
              column: 25
            },
            end: {
              line: 1,
              column: 27
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 27
          }
        }
      },
      alternate: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 27
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 27
    }
  }
});

test("if (morning) goodMorning(); else goodDay()", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Identifier",
        name: "morning",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      consequent: {
        type: "ExpressionStatement",
        expression: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "goodMorning",
            loc: {
              start: {
                line: 1,
                column: 13
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 26
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 27
          }
        }
      },
      alternate: {
        type: "ExpressionStatement",
        expression: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "goodDay",
            loc: {
              start: {
                line: 1,
                column: 33
              },
              end: {
                line: 1,
                column: 40
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 33
            },
            end: {
              line: 1,
              column: 42
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 33
          },
          end: {
            line: 1,
            column: 42
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 42
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 42
    }
  }
});

test("do keep(); while (true)", {
  type: "Program",
  body: [
    {
      type: "DoWhileStatement",
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "keep",
            loc: {
              start: {
                line: 1,
                column: 3
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 3
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 3
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 18
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 23
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 23
    }
  }
});

test("do keep(); while (true);", {
  type: "Program",
  body: [
    {
      type: "DoWhileStatement",
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "keep",
            loc: {
              start: {
                line: 1,
                column: 3
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 3
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 3
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 18
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
});

test("do { x++; y--; } while (x < 10)", {
  type: "Program",
  body: [
    {
      type: "DoWhileStatement",
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "UpdateExpression",
              operator: "++",
              prefix: false,
              argument: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 1,
                    column: 5
                  },
                  end: {
                    line: 1,
                    column: 6
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 5
                },
                end: {
                  line: 1,
                  column: 8
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "UpdateExpression",
              operator: "--",
              prefix: false,
              argument: {
                type: "Identifier",
                name: "y",
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 11
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 10
                },
                end: {
                  line: 1,
                  column: 13
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 10
              },
              end: {
                line: 1,
                column: 14
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 3
          },
          end: {
            line: 1,
            column: 16
          }
        }
      },
      test: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 24
            },
            end: {
              line: 1,
              column: 25
            }
          }
        },
        operator: "<",
        right: {
          type: "Literal",
          value: 10,
          loc: {
            start: {
              line: 1,
              column: 28
            },
            end: {
              line: 1,
              column: 30
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 24
          },
          end: {
            line: 1,
            column: 30
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 31
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 31
    }
  }
});

test("{ do { } while (false);false }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "DoWhileStatement",
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {
                line: 1,
                column: 5
              },
              end: {
                line: 1,
                column: 8
              }
            }
          },
          test: {
            type: "Literal",
            value: false,
            loc: {
              start: {
                line: 1,
                column: 16
              },
              end: {
                line: 1,
                column: 21
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 23
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Literal",
            value: false,
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 28
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 23
            },
            end: {
              line: 1,
              column: 28
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 30
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 30
    }
  }
});

test("while (true) doSomething()", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "CallExpression",
          callee: {
            type: "Identifier",
            name: "doSomething",
            loc: {
              start: {
                line: 1,
                column: 13
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          arguments: [],
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 26
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 26
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 26
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 26
    }
  }
});

test("while (x < 10) { x++; y--; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 8
            }
          }
        },
        operator: "<",
        right: {
          type: "Literal",
          value: 10,
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "UpdateExpression",
              operator: "++",
              prefix: false,
              argument: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 1,
                    column: 17
                  },
                  end: {
                    line: 1,
                    column: 18
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 17
                },
                end: {
                  line: 1,
                  column: 20
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 17
              },
              end: {
                line: 1,
                column: 21
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "UpdateExpression",
              operator: "--",
              prefix: false,
              argument: {
                type: "Identifier",
                name: "y",
                loc: {
                  start: {
                    line: 1,
                    column: 22
                  },
                  end: {
                    line: 1,
                    column: 23
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 22
                },
                end: {
                  line: 1,
                  column: 25
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 22
              },
              end: {
                line: 1,
                column: 26
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 15
          },
          end: {
            line: 1,
            column: 28
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 28
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 28
    }
  }
});

test("for(;;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: null,
      test: null,
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 8
          }
        }
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
  ],
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
});

test("for(;;){}", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: null,
      test: null,
      update: null,
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("for(x = 0;;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        right: {
          type: "Literal",
          value: 0,
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      test: null,
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 12
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("for(var x = 0;;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 8
                },
                end: {
                  line: 1,
                  column: 9
                }
              }
            },
            init: {
              type: "Literal",
              value: 0,
              loc: {
                start: {
                  line: 1,
                  column: 12
                },
                end: {
                  line: 1,
                  column: 13
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          }
        ],
        kind: "var",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      test: null,
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 16
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
});

test("for(var x = 0, y = 1;;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 8
                },
                end: {
                  line: 1,
                  column: 9
                }
              }
            },
            init: {
              type: "Literal",
              value: 0,
              loc: {
                start: {
                  line: 1,
                  column: 12
                },
                end: {
                  line: 1,
                  column: 13
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "y",
              loc: {
                start: {
                  line: 1,
                  column: 15
                },
                end: {
                  line: 1,
                  column: 16
                }
              }
            },
            init: {
              type: "Literal",
              value: 1,
              loc: {
                start: {
                  line: 1,
                  column: 19
                },
                end: {
                  line: 1,
                  column: 20
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 20
              }
            }
          }
        ],
        kind: "var",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 20
          }
        }
      },
      test: null,
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 23
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
});

test("for(x = 0; x < 42;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        right: {
          type: "Literal",
          value: 0,
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      test: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        operator: "<",
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 11
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 19
          },
          end: {
            line: 1,
            column: 20
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 20
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 20
    }
  }
});

test("for(x = 0; x < 42; x++);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        right: {
          type: "Literal",
          value: 0,
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      test: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        operator: "<",
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 11
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      update: {
        type: "UpdateExpression",
        operator: "++",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 20
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 19
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 23
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
});

test("for(x = 0; x < 42; x++) process(x);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        right: {
          type: "Literal",
          value: 0,
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 9
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      test: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        operator: "<",
        right: {
          type: "Literal",
          value: 42,
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 11
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      update: {
        type: "UpdateExpression",
        operator: "++",
        prefix: false,
        argument: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 20
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 19
          },
          end: {
            line: 1,
            column: 22
          }
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
              start: {
                line: 1,
                column: 24
              },
              end: {
                line: 1,
                column: 31
              }
            }
          },
          arguments: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 32
                },
                end: {
                  line: 1,
                  column: 33
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 24
            },
            end: {
              line: 1,
              column: 34
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 24
          },
          end: {
            line: 1,
            column: 35
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 35
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 35
    }
  }
});

test("for(x in list) process(x);", {
  type: "Program",
  body: [
    {
      type: "ForInStatement",
      left: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      right: {
        type: "Identifier",
        name: "list",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 13
          }
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
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 22
              }
            }
          },
          arguments: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 23
                },
                end: {
                  line: 1,
                  column: 24
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 25
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 15
          },
          end: {
            line: 1,
            column: 26
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 26
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 26
    }
  }
});

test("for (var x in list) process(x);", {
  type: "Program",
  body: [
    {
      type: "ForInStatement",
      left: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            init: null,
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 10
              }
            }
          }
        ],
        kind: "var",
        loc: {
          start: {
            line: 1,
            column: 5
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      right: {
        type: "Identifier",
        name: "list",
        loc: {
          start: {
            line: 1,
            column: 14
          },
          end: {
            line: 1,
            column: 18
          }
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
              start: {
                line: 1,
                column: 20
              },
              end: {
                line: 1,
                column: 27
              }
            }
          },
          arguments: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 28
                },
                end: {
                  line: 1,
                  column: 29
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 20
            },
            end: {
              line: 1,
              column: 30
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 20
          },
          end: {
            line: 1,
            column: 31
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 31
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 31
    }
  }
});

test("for (var x = 42 in list) process(x);", {
  type: "Program",
  body: [
    {
      type: "ForInStatement",
      left: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            init: {
              type: "Literal",
              value: 42,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 15
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 15
              }
            }
          }
        ],
        kind: "var",
        loc: {
          start: {
            line: 1,
            column: 5
          },
          end: {
            line: 1,
            column: 15
          }
        }
      },
      right: {
        type: "Identifier",
        name: "list",
        loc: {
          start: {
            line: 1,
            column: 19
          },
          end: {
            line: 1,
            column: 23
          }
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
              start: {
                line: 1,
                column: 25
              },
              end: {
                line: 1,
                column: 32
              }
            }
          },
          arguments: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 33
                },
                end: {
                  line: 1,
                  column: 34
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 25
            },
            end: {
              line: 1,
              column: 35
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 25
          },
          end: {
            line: 1,
            column: 36
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 36
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 36
    }
  }
});

test("for (var i = function() { return 10 in [] } in list) process(x);", {
  type: "Program",
  body: [
    {
      type: "ForInStatement",
      left: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "i",
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            init: {
              type: "FunctionExpression",
              id: null,
              params: [],
              body: {
                type: "BlockStatement",
                body: [
                  {
                    type: "ReturnStatement",
                    argument: {
                      type: "BinaryExpression",
                      left: {
                        type: "Literal",
                        value: 10,
                        loc: {
                          start: {
                            line: 1,
                            column: 33
                          },
                          end: {
                            line: 1,
                            column: 35
                          }
                        }
                      },
                      operator: "in",
                      right: {
                        type: "ArrayExpression",
                        elements: [],
                        loc: {
                          start: {
                            line: 1,
                            column: 39
                          },
                          end: {
                            line: 1,
                            column: 41
                          }
                        }
                      },
                      loc: {
                        start: {
                          line: 1,
                          column: 33
                        },
                        end: {
                          line: 1,
                          column: 41
                        }
                      }
                    },
                    loc: {
                      start: {
                        line: 1,
                        column: 26
                      },
                      end: {
                        line: 1,
                        column: 41
                      }
                    }
                  }
                ],
                loc: {
                  start: {
                    line: 1,
                    column: 24
                  },
                  end: {
                    line: 1,
                    column: 43
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 43
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 43
              }
            }
          }
        ],
        kind: "var",
        loc: {
          start: {
            line: 1,
            column: 5
          },
          end: {
            line: 1,
            column: 43
          }
        }
      },
      right: {
        type: "Identifier",
        name: "list",
        loc: {
          start: {
            line: 1,
            column: 47
          },
          end: {
            line: 1,
            column: 51
          }
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
              start: {
                line: 1,
                column: 53
              },
              end: {
                line: 1,
                column: 60
              }
            }
          },
          arguments: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 61
                },
                end: {
                  line: 1,
                  column: 62
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 53
            },
            end: {
              line: 1,
              column: 63
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 53
          },
          end: {
            line: 1,
            column: 64
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 64
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 64
    }
  }
});

test("while (true) { continue; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ContinueStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 24
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 26
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 26
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 26
    }
  }
});

test("while (true) { continue }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ContinueStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 23
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 25
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 25
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 25
    }
  }
});

test("done: while (true) { continue done }", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "WhileStatement",
        test: {
          type: "Literal",
          value: true,
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ContinueStatement",
              label: {
                type: "Identifier",
                name: "done",
                loc: {
                  start: {
                    line: 1,
                    column: 30
                  },
                  end: {
                    line: 1,
                    column: 34
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 34
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 36
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 36
          }
        }
      },
      label: {
        type: "Identifier",
        name: "done",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 36
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 36
    }
  }
});

test("done: while (true) { continue done; }", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "WhileStatement",
        test: {
          type: "Literal",
          value: true,
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ContinueStatement",
              label: {
                type: "Identifier",
                name: "done",
                loc: {
                  start: {
                    line: 1,
                    column: 30
                  },
                  end: {
                    line: 1,
                    column: 34
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 35
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 37
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 37
          }
        }
      },
      label: {
        type: "Identifier",
        name: "done",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 37
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 37
    }
  }
});

test("while (true) { break }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "BreakStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 20
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("done: while (true) { break done }", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "WhileStatement",
        test: {
          type: "Literal",
          value: true,
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "BreakStatement",
              label: {
                type: "Identifier",
                name: "done",
                loc: {
                  start: {
                    line: 1,
                    column: 27
                  },
                  end: {
                    line: 1,
                    column: 31
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 31
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 33
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 33
          }
        }
      },
      label: {
        type: "Identifier",
        name: "done",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 33
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 33
    }
  }
});

test("done: while (true) { break done; }", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "WhileStatement",
        test: {
          type: "Literal",
          value: true,
          loc: {
            start: {
              line: 1,
              column: 13
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "BreakStatement",
              label: {
                type: "Identifier",
                name: "done",
                loc: {
                  start: {
                    line: 1,
                    column: 27
                  },
                  end: {
                    line: 1,
                    column: 31
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 32
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 34
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 34
          }
        }
      },
      label: {
        type: "Identifier",
        name: "done",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 34
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 34
    }
  }
});

test("(function(){ return })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: null,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 19
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 21
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 1,
            column: 21
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("(function(){ return; })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: null,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 20
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 22
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 23
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 23
    }
  }
});

test("(function(){ return x; })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 1,
                    column: 20
                  },
                  end: {
                    line: 1,
                    column: 21
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 22
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 24
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 25
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 25
    }
  }
});

test("(function(){ return x * y })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: {
                type: "BinaryExpression",
                left: {
                  type: "Identifier",
                  name: "x",
                  loc: {
                    start: {
                      line: 1,
                      column: 20
                    },
                    end: {
                      line: 1,
                      column: 21
                    }
                  }
                },
                operator: "*",
                right: {
                  type: "Identifier",
                  name: "y",
                  loc: {
                    start: {
                      line: 1,
                      column: 24
                    },
                    end: {
                      line: 1,
                      column: 25
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 20
                  },
                  end: {
                    line: 1,
                    column: 25
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 25
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 27
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 1,
            column: 27
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 28
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 28
    }
  }
});

test("with (x) foo = bar", {
  type: "Program",
  body: [
    {
      type: "WithStatement",
      object: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "AssignmentExpression",
          operator: "=",
          left: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 12
              }
            }
          },
          right: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 18
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 18
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 18
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 18
    }
  }
});

test("with (x) foo = bar;", {
  type: "Program",
  body: [
    {
      type: "WithStatement",
      object: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "AssignmentExpression",
          operator: "=",
          left: {
            type: "Identifier",
            name: "foo",
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 12
              }
            }
          },
          right: {
            type: "Identifier",
            name: "bar",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 18
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 18
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 19
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 19
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 19
    }
  }
});

test("with (x) { foo = bar }", {
  type: "Program",
  body: [
    {
      type: "WithStatement",
      object: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "AssignmentExpression",
              operator: "=",
              left: {
                type: "Identifier",
                name: "foo",
                loc: {
                  start: {
                    line: 1,
                    column: 11
                  },
                  end: {
                    line: 1,
                    column: 14
                  }
                }
              },
              right: {
                type: "Identifier",
                name: "bar",
                loc: {
                  start: {
                    line: 1,
                    column: 17
                  },
                  end: {
                    line: 1,
                    column: 20
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 11
                },
                end: {
                  line: 1,
                  column: 20
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 11
              },
              end: {
                line: 1,
                column: 20
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 22
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 22
    }
  }
});

test("switch (x) {}", {
  type: "Program",
  body: [
    {
      type: "SwitchStatement",
      discriminant: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 8
          },
          end: {
            line: 1,
            column: 9
          }
        }
      },
      cases: [],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 13
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 13
    }
  }
});

test("switch (answer) { case 42: hi(); break; }", {
  type: "Program",
  body: [
    {
      type: "SwitchStatement",
      discriminant: {
        type: "Identifier",
        name: "answer",
        loc: {
          start: {
            line: 1,
            column: 8
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      cases: [
        {
          type: "SwitchCase",
          consequent: [
            {
              type: "ExpressionStatement",
              expression: {
                type: "CallExpression",
                callee: {
                  type: "Identifier",
                  name: "hi",
                  loc: {
                    start: {
                      line: 1,
                      column: 27
                    },
                    end: {
                      line: 1,
                      column: 29
                    }
                  }
                },
                arguments: [],
                loc: {
                  start: {
                    line: 1,
                    column: 27
                  },
                  end: {
                    line: 1,
                    column: 31
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 27
                },
                end: {
                  line: 1,
                  column: 32
                }
              }
            },
            {
              type: "BreakStatement",
              label: null,
              loc: {
                start: {
                  line: 1,
                  column: 33
                },
                end: {
                  line: 1,
                  column: 39
                }
              }
            }
          ],
          test: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 25
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 39
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 41
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 41
    }
  }
});

test("switch (answer) { case 42: hi(); break; default: break }", {
  type: "Program",
  body: [
    {
      type: "SwitchStatement",
      discriminant: {
        type: "Identifier",
        name: "answer",
        loc: {
          start: {
            line: 1,
            column: 8
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      cases: [
        {
          type: "SwitchCase",
          consequent: [
            {
              type: "ExpressionStatement",
              expression: {
                type: "CallExpression",
                callee: {
                  type: "Identifier",
                  name: "hi",
                  loc: {
                    start: {
                      line: 1,
                      column: 27
                    },
                    end: {
                      line: 1,
                      column: 29
                    }
                  }
                },
                arguments: [],
                loc: {
                  start: {
                    line: 1,
                    column: 27
                  },
                  end: {
                    line: 1,
                    column: 31
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 27
                },
                end: {
                  line: 1,
                  column: 32
                }
              }
            },
            {
              type: "BreakStatement",
              label: null,
              loc: {
                start: {
                  line: 1,
                  column: 33
                },
                end: {
                  line: 1,
                  column: 39
                }
              }
            }
          ],
          test: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 25
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 39
            }
          }
        },
        {
          type: "SwitchCase",
          consequent: [
            {
              type: "BreakStatement",
              label: null,
              loc: {
                start: {
                  line: 1,
                  column: 49
                },
                end: {
                  line: 1,
                  column: 54
                }
              }
            }
          ],
          test: null,
          loc: {
            start: {
              line: 1,
              column: 40
            },
            end: {
              line: 1,
              column: 54
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 56
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 56
    }
  }
});

test("start: for (;;) break start", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "ForStatement",
        init: null,
        test: null,
        update: null,
        body: {
          type: "BreakStatement",
          label: {
            type: "Identifier",
            name: "start",
            loc: {
              start: {
                line: 1,
                column: 22
              },
              end: {
                line: 1,
                column: 27
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 16
            },
            end: {
              line: 1,
              column: 27
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 27
          }
        }
      },
      label: {
        type: "Identifier",
        name: "start",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 27
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 27
    }
  }
});

test("start: while (true) break start", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "WhileStatement",
        test: {
          type: "Literal",
          value: true,
          loc: {
            start: {
              line: 1,
              column: 14
            },
            end: {
              line: 1,
              column: 18
            }
          }
        },
        body: {
          type: "BreakStatement",
          label: {
            type: "Identifier",
            name: "start",
            loc: {
              start: {
                line: 1,
                column: 26
              },
              end: {
                line: 1,
                column: 31
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 20
            },
            end: {
              line: 1,
              column: 31
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 31
          }
        }
      },
      label: {
        type: "Identifier",
        name: "start",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 31
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 31
    }
  }
});

test("throw x;", {
  type: "Program",
  body: [
    {
      type: "ThrowStatement",
      argument: {
        type: "Identifier",
        name: "x",
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 7
          }
        }
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
  ],
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
});

test("throw x * y", {
  type: "Program",
  body: [
    {
      type: "ThrowStatement",
      argument: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "x",
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 7
            }
          }
        },
        operator: "*",
        right: {
          type: "Identifier",
          name: "y",
          loc: {
            start: {
              line: 1,
              column: 10
            },
            end: {
              line: 1,
              column: 11
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 11
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 11
    }
  }
});

test("throw { message: \"Error\" }", {
  type: "Program",
  body: [
    {
      type: "ThrowStatement",
      argument: {
        type: "ObjectExpression",
        properties: [
          {
            type: "Property",
            key: {
              type: "Identifier",
              name: "message",
              loc: {
                start: {
                  line: 1,
                  column: 8
                },
                end: {
                  line: 1,
                  column: 15
                }
              }
            },
            value: {
              type: "Literal",
              value: "Error",
              loc: {
                start: {
                  line: 1,
                  column: 17
                },
                end: {
                  line: 1,
                  column: 24
                }
              }
            },
            kind: "init"
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 6
          },
          end: {
            line: 1,
            column: 26
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 26
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 26
    }
  }
});

test("try { } catch (e) { }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      handler: {
          type: "CatchClause",
          param: {
            type: "Identifier",
            name: "e",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 16
              }
            }
          },
          guard: null,
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 21
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 21
            }
          }
        }
      ,
      finalizer: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 21
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 21
    }
  }
});

test("try { } catch (eval) { }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      handler:
        {
          type: "CatchClause",
          param: {
            type: "Identifier",
            name: "eval",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 19
              }
            }
          },
          guard: null,
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {
                line: 1,
                column: 21
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 24
            }
          }
        }
      ,
      finalizer: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
});

test("try { } catch (arguments) { }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      handler:
        {
          type: "CatchClause",
          param: {
            type: "Identifier",
            name: "arguments",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          guard: null,
          body: {
            type: "BlockStatement",
            body: [],
            loc: {
              start: {
                line: 1,
                column: 26
              },
              end: {
                line: 1,
                column: 29
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 29
            }
          }
        }
      ,
      finalizer: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
});

test("try { } catch (e) { say(e) }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      handler:
        {
          type: "CatchClause",
          param: {
            type: "Identifier",
            name: "e",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 16
              }
            }
          },
          guard: null,
          body: {
            type: "BlockStatement",
            body: [
              {
                type: "ExpressionStatement",
                expression: {
                  type: "CallExpression",
                  callee: {
                    type: "Identifier",
                    name: "say",
                    loc: {
                      start: {
                        line: 1,
                        column: 20
                      },
                      end: {
                        line: 1,
                        column: 23
                      }
                    }
                  },
                  arguments: [
                    {
                      type: "Identifier",
                      name: "e",
                      loc: {
                        start: {
                          line: 1,
                          column: 24
                        },
                        end: {
                          line: 1,
                          column: 25
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 20
                    },
                    end: {
                      line: 1,
                      column: 26
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 20
                  },
                  end: {
                    line: 1,
                    column: 26
                  }
                }
              }
            ],
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 28
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 8
            },
            end: {
              line: 1,
              column: 28
            }
          }
        }
      ,
      finalizer: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 28
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 28
    }
  }
});

test("try { } finally { cleanup(stuff) }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 7
          }
        }
      },
      handler: null,
      finalizer: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "cleanup",
                loc: {
                  start: {
                    line: 1,
                    column: 18
                  },
                  end: {
                    line: 1,
                    column: 25
                  }
                }
              },
              arguments: [
                {
                  type: "Identifier",
                  name: "stuff",
                  loc: {
                    start: {
                      line: 1,
                      column: 26
                    },
                    end: {
                      line: 1,
                      column: 31
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 18
                },
                end: {
                  line: 1,
                  column: 32
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 32
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 16
          },
          end: {
            line: 1,
            column: 34
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 34
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 34
    }
  }
});

test("try { doThat(); } catch (e) { say(e) }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "doThat",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              arguments: [],
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 14
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 6
              },
              end: {
                line: 1,
                column: 15
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      handler:
        {
          type: "CatchClause",
          param: {
            type: "Identifier",
            name: "e",
            loc: {
              start: {
                line: 1,
                column: 25
              },
              end: {
                line: 1,
                column: 26
              }
            }
          },
          guard: null,
          body: {
            type: "BlockStatement",
            body: [
              {
                type: "ExpressionStatement",
                expression: {
                  type: "CallExpression",
                  callee: {
                    type: "Identifier",
                    name: "say",
                    loc: {
                      start: {
                        line: 1,
                        column: 30
                      },
                      end: {
                        line: 1,
                        column: 33
                      }
                    }
                  },
                  arguments: [
                    {
                      type: "Identifier",
                      name: "e",
                      loc: {
                        start: {
                          line: 1,
                          column: 34
                        },
                        end: {
                          line: 1,
                          column: 35
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 30
                    },
                    end: {
                      line: 1,
                      column: 36
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 30
                  },
                  end: {
                    line: 1,
                    column: 36
                  }
                }
              }
            ],
            loc: {
              start: {
                line: 1,
                column: 28
              },
              end: {
                line: 1,
                column: 38
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 38
            }
          }
        }
      ,
      finalizer: null,
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 38
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 38
    }
  }
});

test("try { doThat(); } catch (e) { say(e) } finally { cleanup(stuff) }", {
  type: "Program",
  body: [
    {
      type: "TryStatement",
      block: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "doThat",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              arguments: [],
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 14
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 6
              },
              end: {
                line: 1,
                column: 15
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      handler:
        {
          type: "CatchClause",
          param: {
            type: "Identifier",
            name: "e",
            loc: {
              start: {
                line: 1,
                column: 25
              },
              end: {
                line: 1,
                column: 26
              }
            }
          },
          guard: null,
          body: {
            type: "BlockStatement",
            body: [
              {
                type: "ExpressionStatement",
                expression: {
                  type: "CallExpression",
                  callee: {
                    type: "Identifier",
                    name: "say",
                    loc: {
                      start: {
                        line: 1,
                        column: 30
                      },
                      end: {
                        line: 1,
                        column: 33
                      }
                    }
                  },
                  arguments: [
                    {
                      type: "Identifier",
                      name: "e",
                      loc: {
                        start: {
                          line: 1,
                          column: 34
                        },
                        end: {
                          line: 1,
                          column: 35
                        }
                      }
                    }
                  ],
                  loc: {
                    start: {
                      line: 1,
                      column: 30
                    },
                    end: {
                      line: 1,
                      column: 36
                    }
                  }
                },
                loc: {
                  start: {
                    line: 1,
                    column: 30
                  },
                  end: {
                    line: 1,
                    column: 36
                  }
                }
              }
            ],
            loc: {
              start: {
                line: 1,
                column: 28
              },
              end: {
                line: 1,
                column: 38
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 38
            }
          }
        }
      ,
      finalizer: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "cleanup",
                loc: {
                  start: {
                    line: 1,
                    column: 49
                  },
                  end: {
                    line: 1,
                    column: 56
                  }
                }
              },
              arguments: [
                {
                  type: "Identifier",
                  name: "stuff",
                  loc: {
                    start: {
                      line: 1,
                      column: 57
                    },
                    end: {
                      line: 1,
                      column: 62
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 49
                },
                end: {
                  line: 1,
                  column: 63
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 49
              },
              end: {
                line: 1,
                column: 63
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 47
          },
          end: {
            line: 1,
            column: 65
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 65
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 65
    }
  }
});

test("debugger;", {
  type: "Program",
  body: [
    {
      type: "DebuggerStatement",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
});

test("function hello() { sayHi(); }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "hello",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      params: [],
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "sayHi",
                loc: {
                  start: {
                    line: 1,
                    column: 19
                  },
                  end: {
                    line: 1,
                    column: 24
                  }
                }
              },
              arguments: [],
              loc: {
                start: {
                  line: 1,
                  column: 19
                },
                end: {
                  line: 1,
                  column: 26
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 19
              },
              end: {
                line: 1,
                column: 27
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 17
          },
          end: {
            line: 1,
            column: 29
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
});

test("function eval() { }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "eval",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      params: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 16
          },
          end: {
            line: 1,
            column: 19
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 19
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 19
    }
  }
});

test("function arguments() { }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "arguments",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 18
          }
        }
      },
      params: [],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 21
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
});

test("function test(t, t) { }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "test",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      params: [
        {
          type: "Identifier",
          name: "t",
          loc: {
            start: {
              line: 1,
              column: 14
            },
            end: {
              line: 1,
              column: 15
            }
          }
        },
        {
          type: "Identifier",
          name: "t",
          loc: {
            start: {
              line: 1,
              column: 17
            },
            end: {
              line: 1,
              column: 18
            }
          }
        }
      ],
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 20
          },
          end: {
            line: 1,
            column: 23
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 23
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 23
    }
  }
});

test("(function test(t, t) { })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: {
          type: "Identifier",
          name: "test",
          loc: {
            start: {
              line: 1,
              column: 10
            },
            end: {
              line: 1,
              column: 14
            }
          }
        },
        params: [
          {
            type: "Identifier",
            name: "t",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 16
              }
            }
          },
          {
            type: "Identifier",
            name: "t",
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 19
              }
            }
          }
        ],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {
              line: 1,
              column: 21
            },
            end: {
              line: 1,
              column: 24
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 25
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 25
    }
  }
});

test("function eval() { function inner() { \"use strict\" } }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "eval",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      params: [],
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "FunctionDeclaration",
            id: {
              type: "Identifier",
              name: "inner",
              loc: {
                start: {
                  line: 1,
                  column: 27
                },
                end: {
                  line: 1,
                  column: 32
                }
              }
            },
            params: [],
            body: {
              type: "BlockStatement",
              body: [
                {
                  type: "ExpressionStatement",
                  expression: {
                    type: "Literal",
                    value: "use strict",
                    loc: {
                      start: {
                        line: 1,
                        column: 37
                      },
                      end: {
                        line: 1,
                        column: 49
                      }
                    }
                  },
                  loc: {
                    start: {
                      line: 1,
                      column: 37
                    },
                    end: {
                      line: 1,
                      column: 49
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 35
                },
                end: {
                  line: 1,
                  column: 51
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 51
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 16
          },
          end: {
            line: 1,
            column: 53
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 53
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 53
    }
  }
});

test("function hello(a) { sayHi(); }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "hello",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      params: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 16
            }
          }
        }
      ],
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "sayHi",
                loc: {
                  start: {
                    line: 1,
                    column: 20
                  },
                  end: {
                    line: 1,
                    column: 25
                  }
                }
              },
              arguments: [],
              loc: {
                start: {
                  line: 1,
                  column: 20
                },
                end: {
                  line: 1,
                  column: 27
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 20
              },
              end: {
                line: 1,
                column: 28
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 18
          },
          end: {
            line: 1,
            column: 30
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 30
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 30
    }
  }
});

test("function hello(a, b) { sayHi(); }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "hello",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      params: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 16
            }
          }
        },
        {
          type: "Identifier",
          name: "b",
          loc: {
            start: {
              line: 1,
              column: 18
            },
            end: {
              line: 1,
              column: 19
            }
          }
        }
      ],
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ExpressionStatement",
            expression: {
              type: "CallExpression",
              callee: {
                type: "Identifier",
                name: "sayHi",
                loc: {
                  start: {
                    line: 1,
                    column: 23
                  },
                  end: {
                    line: 1,
                    column: 28
                  }
                }
              },
              arguments: [],
              loc: {
                start: {
                  line: 1,
                  column: 23
                },
                end: {
                  line: 1,
                  column: 30
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 31
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 21
          },
          end: {
            line: 1,
            column: 33
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 33
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 33
    }
  }
});

test("function hello(...rest) { }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "hello",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      params: [],
      rest: {
        type: "Identifier",
        name: "rest",
        loc: {
          start: {
            line: 1,
            column: 18
          },
          end: {
            line: 1,
            column: 22
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 24
          },
          end: {
            line: 1,
            column: 27
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 27
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 27
    }
  }
}, {
  ecmaVersion: 6,
  locations: true
});

test("function hello(a, ...rest) { }", {
  type: "Program",
  body: [
    {
      type: "FunctionDeclaration",
      id: {
        type: "Identifier",
        name: "hello",
        loc: {
          start: {
            line: 1,
            column: 9
          },
          end: {
            line: 1,
            column: 14
          }
        }
      },
      params: [
        {
          type: "Identifier",
          name: "a",
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 16
            }
          }
        }
      ],
      rest: {
        type: "Identifier",
        name: "rest",
        loc: {
          start: {
            line: 1,
            column: 21
          },
          end: {
            line: 1,
            column: 25
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [],
        loc: {
          start: {
            line: 1,
            column: 27
          },
          end: {
            line: 1,
            column: 30
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 30
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 30
    }
  }
}, {
  ecmaVersion: 6,
  locations: true
});

test("var hi = function() { sayHi() };", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "hi",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          init: {
            type: "FunctionExpression",
            id: null,
            params: [],
            body: {
              type: "BlockStatement",
              body: [
                {
                  type: "ExpressionStatement",
                  expression: {
                    type: "CallExpression",
                    callee: {
                      type: "Identifier",
                      name: "sayHi",
                      loc: {
                        start: {
                          line: 1,
                          column: 22
                        },
                        end: {
                          line: 1,
                          column: 27
                        }
                      }
                    },
                    arguments: [],
                    loc: {
                      start: {
                        line: 1,
                        column: 22
                      },
                      end: {
                        line: 1,
                        column: 29
                      }
                    }
                  },
                  loc: {
                    start: {
                      line: 1,
                      column: 22
                    },
                    end: {
                      line: 1,
                      column: 29
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 20
                },
                end: {
                  line: 1,
                  column: 31
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 31
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 31
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 32
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 32
    }
  }
});

test("var hi = function (...r) { sayHi() };", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "hi",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          init: {
            type: "FunctionExpression",
            id: null,
            params: [],
            rest: {
              type: "Identifier",
              name: "r",
              loc: {
                start: {
                  line: 1,
                  column: 22
                },
                end: {
                  line: 1,
                  column: 23
                }
              }
            },
            body: {
              type: "BlockStatement",
              body: [
                {
                  type: "ExpressionStatement",
                  expression: {
                    type: "CallExpression",
                    callee: {
                      type: "Identifier",
                      name: "sayHi",
                      loc: {
                        start: {
                          line: 1,
                          column: 27
                        },
                        end: {
                          line: 1,
                          column: 32
                        }
                      }
                    },
                    arguments: [],
                    loc: {
                      start: {
                        line: 1,
                        column: 27
                      },
                      end: {
                        line: 1,
                        column: 34
                      }
                    }
                  },
                  loc: {
                    start: {
                      line: 1,
                      column: 27
                    },
                    end: {
                      line: 1,
                      column: 34
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 25
                },
                end: {
                  line: 1,
                  column: 36
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 36
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 36
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 37
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 37
    }
  }
}, {
  ecmaVersion: 6,
  locations: true
});

test("var hi = function eval() { };", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "hi",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          init: {
            type: "FunctionExpression",
            id: {
              type: "Identifier",
              name: "eval",
              loc: {
                start: {
                  line: 1,
                  column: 18
                },
                end: {
                  line: 1,
                  column: 22
                }
              }
            },
            params: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {
                  line: 1,
                  column: 25
                },
                end: {
                  line: 1,
                  column: 28
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 28
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 28
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
});

test("var hi = function arguments() { };", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "hi",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 6
              }
            }
          },
          init: {
            type: "FunctionExpression",
            id: {
              type: "Identifier",
              name: "arguments",
              loc: {
                start: {
                  line: 1,
                  column: 18
                },
                end: {
                  line: 1,
                  column: 27
                }
              }
            },
            params: [],
            body: {
              type: "BlockStatement",
              body: [],
              loc: {
                start: {
                  line: 1,
                  column: 30
                },
                end: {
                  line: 1,
                  column: 33
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 33
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 33
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 34
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 34
    }
  }
});

test("var hello = function hi() { sayHi() };", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "hello",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 9
              }
            }
          },
          init: {
            type: "FunctionExpression",
            id: {
              type: "Identifier",
              name: "hi",
              loc: {
                start: {
                  line: 1,
                  column: 21
                },
                end: {
                  line: 1,
                  column: 23
                }
              }
            },
            params: [],
            body: {
              type: "BlockStatement",
              body: [
                {
                  type: "ExpressionStatement",
                  expression: {
                    type: "CallExpression",
                    callee: {
                      type: "Identifier",
                      name: "sayHi",
                      loc: {
                        start: {
                          line: 1,
                          column: 28
                        },
                        end: {
                          line: 1,
                          column: 33
                        }
                      }
                    },
                    arguments: [],
                    loc: {
                      start: {
                        line: 1,
                        column: 28
                      },
                      end: {
                        line: 1,
                        column: 35
                      }
                    }
                  },
                  loc: {
                    start: {
                      line: 1,
                      column: 28
                    },
                    end: {
                      line: 1,
                      column: 35
                    }
                  }
                }
              ],
              loc: {
                start: {
                  line: 1,
                  column: 26
                },
                end: {
                  line: 1,
                  column: 37
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 12
              },
              end: {
                line: 1,
                column: 37
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 37
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 38
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 38
    }
  }
});

test("(function(){})", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 14
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 14
    }
  }
});

test("{ x\n++y }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 2
              },
              end: {
                line: 1,
                column: 3
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 3
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "UpdateExpression",
            operator: "++",
            prefix: true,
            argument: {
              type: "Identifier",
              name: "y",
              loc: {
                start: {
                  line: 2,
                  column: 2
                },
                end: {
                  line: 2,
                  column: 3
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 3
              }
            }
          },
          loc: {
            start: {
              line: 2,
              column: 0
            },
            end: {
              line: 2,
              column: 3
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 5
    }
  }
});

test("{ x\n--y }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 2
              },
              end: {
                line: 1,
                column: 3
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 3
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "UpdateExpression",
            operator: "--",
            prefix: true,
            argument: {
              type: "Identifier",
              name: "y",
              loc: {
                start: {
                  line: 2,
                  column: 2
                },
                end: {
                  line: 2,
                  column: 3
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 3
              }
            }
          },
          loc: {
            start: {
              line: 2,
              column: 0
            },
            end: {
              line: 2,
              column: 3
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 5
    }
  }
});

test("var x /* comment */;", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        }
      ],
      kind: "var",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 20
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 20
    }
  }
});

test("{ var x = 14, y = 3\nz; }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "VariableDeclaration",
          declarations: [
            {
              type: "VariableDeclarator",
              id: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 1,
                    column: 6
                  },
                  end: {
                    line: 1,
                    column: 7
                  }
                }
              },
              init: {
                type: "Literal",
                value: 14,
                loc: {
                  start: {
                    line: 1,
                    column: 10
                  },
                  end: {
                    line: 1,
                    column: 12
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 6
                },
                end: {
                  line: 1,
                  column: 12
                }
              }
            },
            {
              type: "VariableDeclarator",
              id: {
                type: "Identifier",
                name: "y",
                loc: {
                  start: {
                    line: 1,
                    column: 14
                  },
                  end: {
                    line: 1,
                    column: 15
                  }
                }
              },
              init: {
                type: "Literal",
                value: 3,
                loc: {
                  start: {
                    line: 1,
                    column: 18
                  },
                  end: {
                    line: 1,
                    column: 19
                  }
                }
              },
              loc: {
                start: {
                  line: 1,
                  column: 14
                },
                end: {
                  line: 1,
                  column: 19
                }
              }
            }
          ],
          kind: "var",
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 19
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 1
              }
            }
          },
          loc: {
            start: {
              line: 2,
              column: 0
            },
            end: {
              line: 2,
              column: 2
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 4
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 4
    }
  }
});

test("while (true) { continue\nthere; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ContinueStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 23
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "there",
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 5
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 6
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 2,
            column: 8
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 8
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 8
    }
  }
});

test("while (true) { continue // Comment\nthere; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ContinueStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 23
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "there",
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 5
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 6
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 2,
            column: 8
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 8
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 8
    }
  }
});

test("while (true) { continue /* Multiline\nComment */there; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "ContinueStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 23
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "there",
              loc: {
                start: {
                  line: 2,
                  column: 10
                },
                end: {
                  line: 2,
                  column: 15
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 10
              },
              end: {
                line: 2,
                column: 16
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 2,
            column: 18
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 18
    }
  }
});

test("while (true) { break\nthere; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "BreakStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "there",
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 5
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 6
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 2,
            column: 8
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 8
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 8
    }
  }
});

test("while (true) { break // Comment\nthere; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "BreakStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "there",
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 5
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 6
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 2,
            column: 8
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 8
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 8
    }
  }
});

test("while (true) { break /* Multiline\nComment */there; }", {
  type: "Program",
  body: [
    {
      type: "WhileStatement",
      test: {
        type: "Literal",
        value: true,
        loc: {
          start: {
            line: 1,
            column: 7
          },
          end: {
            line: 1,
            column: 11
          }
        }
      },
      body: {
        type: "BlockStatement",
        body: [
          {
            type: "BreakStatement",
            label: null,
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          {
            type: "ExpressionStatement",
            expression: {
              type: "Identifier",
              name: "there",
              loc: {
                start: {
                  line: 2,
                  column: 10
                },
                end: {
                  line: 2,
                  column: 15
                }
              }
            },
            loc: {
              start: {
                line: 2,
                column: 10
              },
              end: {
                line: 2,
                column: 16
              }
            }
          }
        ],
        loc: {
          start: {
            line: 1,
            column: 13
          },
          end: {
            line: 2,
            column: 18
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 18
    }
  }
});

test("(function(){ return\nx; })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: null,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 19
                }
              }
            },
            {
              type: "ExpressionStatement",
              expression: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 2,
                    column: 0
                  },
                  end: {
                    line: 2,
                    column: 1
                  }
                }
              },
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 2
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 2,
              column: 4
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 2,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 5
    }
  }
});

test("(function(){ return // Comment\nx; })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: null,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 19
                }
              }
            },
            {
              type: "ExpressionStatement",
              expression: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 2,
                    column: 0
                  },
                  end: {
                    line: 2,
                    column: 1
                  }
                }
              },
              loc: {
                start: {
                  line: 2,
                  column: 0
                },
                end: {
                  line: 2,
                  column: 2
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 2,
              column: 4
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 2,
            column: 4
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 5
    }
  }
});

test("(function(){ return/* Multiline\nComment */x; })", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "FunctionExpression",
        id: null,
        params: [],
        body: {
          type: "BlockStatement",
          body: [
            {
              type: "ReturnStatement",
              argument: null,
              loc: {
                start: {
                  line: 1,
                  column: 13
                },
                end: {
                  line: 1,
                  column: 19
                }
              }
            },
            {
              type: "ExpressionStatement",
              expression: {
                type: "Identifier",
                name: "x",
                loc: {
                  start: {
                    line: 2,
                    column: 10
                  },
                  end: {
                    line: 2,
                    column: 11
                  }
                }
              },
              loc: {
                start: {
                  line: 2,
                  column: 10
                },
                end: {
                  line: 2,
                  column: 12
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 11
            },
            end: {
              line: 2,
              column: 14
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 2,
            column: 14
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 15
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 15
    }
  }
});

test("{ throw error\nerror; }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ThrowStatement",
          argument: {
            type: "Identifier",
            name: "error",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "error",
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 2,
              column: 0
            },
            end: {
              line: 2,
              column: 6
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 8
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 8
    }
  }
});

test("{ throw error// Comment\nerror; }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ThrowStatement",
          argument: {
            type: "Identifier",
            name: "error",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "error",
            loc: {
              start: {
                line: 2,
                column: 0
              },
              end: {
                line: 2,
                column: 5
              }
            }
          },
          loc: {
            start: {
              line: 2,
              column: 0
            },
            end: {
              line: 2,
              column: 6
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 8
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 8
    }
  }
});

test("{ throw error/* Multiline\nComment */error; }", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: [
        {
          type: "ThrowStatement",
          argument: {
            type: "Identifier",
            name: "error",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 2
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        {
          type: "ExpressionStatement",
          expression: {
            type: "Identifier",
            name: "error",
            loc: {
              start: {
                line: 2,
                column: 10
              },
              end: {
                line: 2,
                column: 15
              }
            }
          },
          loc: {
            start: {
              line: 2,
              column: 10
            },
            end: {
              line: 2,
              column: 16
            }
          }
        }
      ],
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 2,
          column: 18
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 2,
      column: 18
    }
  }
});

test("", {
  type: "Program",
  body: [],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 0
    }
  }
});

test("foo: if (true) break foo;", {
  type: "Program",
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 25
    }
  },
  body: [
    {
      type: "LabeledStatement",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 25
        }
      },
      body: {
        type: "IfStatement",
        loc: {
          start: {
            line: 1,
            column: 5
          },
          end: {
            line: 1,
            column: 25
          }
        },
        test: {
          type: "Literal",
          loc: {
            start: {
              line: 1,
              column: 9
            },
            end: {
              line: 1,
              column: 13
            }
          },
          value: true
        },
        consequent: {
          type: "BreakStatement",
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 25
            }
          },
          label: {
            type: "Identifier",
            loc: {
              start: {
                line: 1,
                column: 21
              },
              end: {
                line: 1,
                column: 24
              }
            },
            name: "foo"
          }
        },
        alternate: null
      },
      label: {
        type: "Identifier",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 3
          }
        },
        name: "foo"
      }
    }
  ]
});

test("(function () {\n 'use strict';\n '\0';\n}())", {
  type: "Program",
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 4,
      column: 4
    }
  },
  body: [
    {
      type: "ExpressionStatement",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 4,
          column: 4
        }
      },
      expression: {
        type: "CallExpression",
        loc: {
          start: {
            line: 1,
            column: 1
          },
          end: {
            line: 4,
            column: 3
          }
        },
        callee: {
          type: "FunctionExpression",
          loc: {
            start: {
              line: 1,
              column: 1
            },
            end: {
              line: 4,
              column: 1
            }
          },
          id: null,
          params: [],
          body: {
            type: "BlockStatement",
            loc: {
              start: {
                line: 1,
                column: 13
              },
              end: {
                line: 4,
                column: 1
              }
            },
            body: [
              {
                type: "ExpressionStatement",
                loc: {
                  start: {
                    line: 2,
                    column: 1
                  },
                  end: {
                    line: 2,
                    column: 14
                  }
                },
                expression: {
                  type: "Literal",
                  loc: {
                    start: {
                      line: 2,
                      column: 1
                    },
                    end: {
                      line: 2,
                      column: 13
                    }
                  },
                  value: "use strict"
                }
              },
              {
                type: "ExpressionStatement",
                loc: {
                  start: {
                    line: 3,
                    column: 1
                  },
                  end: {
                    line: 3,
                    column: 5
                  }
                },
                expression: {
                  type: "Literal",
                  loc: {
                    start: {
                      line: 3,
                      column: 1
                    },
                    end: {
                      line: 3,
                      column: 4
                    }
                  },
                  value: "\u0000"
                }
              }
            ]
          }
        },
        arguments: [],
      }
    }
  ]
});

test("123..toString(10)", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "Literal",
            value: 123
          },
          property: {
            type: "Identifier",
            name: "toString"
          },
          computed: false,
        },
        arguments: [
          {
            type: "Literal",
            value: 10
          }
        ],
      }
    }
  ]
});

test("123.+2", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Literal",
          value: 123
        },
        operator: "+",
        right: {
          type: "Literal",
          value: 2
        },
      }
    }
  ]
});

test("a\u2028b", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Identifier",
        name: "a"
      }
    },
    {
      type: "ExpressionStatement",
      expression: {
        type: "Identifier",
        name: "b"
      }
    }
  ]
});

test("'a\\u0026b'", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "a\u0026b"
      }
    }
  ]
});

test("foo: 10; foo: 20;", {
  type: "Program",
  body: [
    {
      type: "LabeledStatement",
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "Literal",
          value: 10,
          raw: "10"
        }
      },
      label: {
        type: "Identifier",
        name: "foo"
      }
    },
    {
      type: "LabeledStatement",
      body: {
        type: "ExpressionStatement",
        expression: {
          type: "Literal",
          value: 20,
          raw: "20"
        }
      },
      label: {
        type: "Identifier",
        name: "foo"
      }
    }
  ]
});

test("if(1)/  foo/", {
  type: "Program",
  body: [
    {
      type: "IfStatement",
      test: {
        type: "Literal",
        value: 1,
        raw: "1"
      },
      consequent: {
        type: "ExpressionStatement",
        expression: {
          type: "Literal",
          raw: "/  foo/"
        }
      },
      alternate: null
    }
  ]
});

test("price_9̶9̶_89", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Identifier",
        name: "price_9̶9̶_89",
      }
    }
  ]
});

// option tests

test("var a = 1;", {
  type: "Program",
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 10
    },
    source: "test.js"
  },
  body: [
    {
      type: "VariableDeclaration",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 10
        },
        source: "test.js"
      },
      declarations: [
        {
          type: "VariableDeclarator",
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 9
            },
            source: "test.js"
          },
          id: {
            type: "Identifier",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              },
              source: "test.js"
            },
            name: "a"
          },
          init: {
            type: "Literal",
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 9
              },
              source: "test.js"
            },
            value: 1,
            raw: "1"
          }
        }
      ],
      kind: "var"
    }
  ]
}, {
  locations: true,
  sourceFile: "test.js"
});

test("a.in / b", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "MemberExpression",
          object: {
            type: "Identifier",
            name: "a"
          },
          property: {
            type: "Identifier",
            name: "in"
          },
          computed: false
        },
        operator: "/",
        right: {
          type: "Identifier",
          name: "b"
        }
      }
    }
  ]
});

test("{}/=/", {
  type: "Program",
  body: [
    {
      type: "BlockStatement",
      body: []
    },
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        raw: "/=/"
      }
    }
  ]
});

test("foo <!--bar\n+baz", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "BinaryExpression",
        left: {
          type: "Identifier",
          name: "foo"
        },
        operator: "+",
        right: {
          type: "Identifier",
          name: "baz"
        }
      }
    }
  ]
});

test("x = y-->10;\n --> nothing", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "AssignmentExpression",
        operator: "=",
        left: {
          type: "Identifier",
          name: "x"
        },
        right: {
          type: "BinaryExpression",
          left: {
            type: "UpdateExpression",
            operator: "--",
            prefix: false,
            argument: {
              type: "Identifier",
              name: "y"
            }
          },
          operator: ">",
          right: {
            type: "Literal",
            value: 10
          }
        }
      }
    }
  ]
});

test("'use strict';\nobject.static();", {
  type: "Program",
  body: [
    {
      type: "ExpressionStatement",
      expression: {
        type: "Literal",
        value: "use strict",
        raw: "'use strict'"
      }
    },
    {
      type: "ExpressionStatement",
      expression: {
        type: "CallExpression",
        callee: {
          type: "MemberExpression",
          object: {
            type: "Identifier",
            name: "object"
          },
          property: {
            type: "Identifier",
            name: "static"
          },
          computed: false
        },
        arguments: []
      }
    }
  ]
});

// Failure tests

testFail("{",
         "Unexpected token (1:1)");

testFail("}",
         "Unexpected token (1:0)");

testFail("3ea",
         "Invalid number (1:0)");

testFail("3in []",
         "Identifier directly after number (1:1)");

testFail("3e",
         "Invalid number (1:0)");

testFail("3e+",
         "Invalid number (1:0)");

testFail("3e-",
         "Invalid number (1:0)");

testFail("3x",
         "Identifier directly after number (1:1)");

testFail("3x0",
         "Identifier directly after number (1:1)");

testFail("0x",
         "Expected number in radix 16 (1:2)");

testFail("09",
         "Invalid number (1:0)");

testFail("018",
         "Invalid number (1:0)");

testFail("01a",
         "Identifier directly after number (1:2)");

testFail("3in[]",
         "Identifier directly after number (1:1)");

testFail("0x3in[]",
         "Identifier directly after number (1:3)");

testFail("\"Hello\nWorld\"",
         "Unterminated string constant (1:0)");

testFail("x\\",
         "Expecting Unicode escape sequence \\uXXXX (1:2)");

testFail("x\\u005c",
         "Invalid Unicode escape (1:3)");

testFail("x\\u002a",
         "Invalid Unicode escape (1:3)");

testFail("/",
         "Unterminated regular expression (1:1)");

testFail("/test",
         "Unterminated regular expression (1:1)");

testFail("var x = /[a-z]/\\ux",
         "Bad character escape sequence (1:8)");

testFail("3 = 4",
         "Assigning to rvalue (1:0)");

testFail("func() = 4",
         "Assigning to rvalue (1:0)");

testFail("(1 + 1) = 10",
         "Assigning to rvalue (1:1)");

testFail("1++",
         "Assigning to rvalue (1:0)");

testFail("1--",
         "Assigning to rvalue (1:0)");

testFail("++1",
         "Assigning to rvalue (1:2)");

testFail("--1",
         "Assigning to rvalue (1:2)");

testFail("for((1 + 1) in list) process(x);",
         "Assigning to rvalue (1:5)");

testFail("[",
         "Unexpected token (1:1)");

testFail("[,",
         "Unexpected token (1:2)");

testFail("1 + {",
         "Unexpected token (1:5)");

testFail("1 + { t:t ",
         "Unexpected token (1:10)");

testFail("1 + { t:t,",
         "Unexpected token (1:10)");

testFail("var x = /\n/",
         "Unterminated regular expression (1:9)");

testFail("var x = \"\n",
         "Unterminated string constant (1:8)");

testFail("var if = 42",
         "Unexpected token (1:4)");

testFail("i + 2 = 42",
         "Assigning to rvalue (1:0)");

testFail("+i = 42",
         "Assigning to rvalue (1:0)");

testFail("1 + (",
         "Unexpected token (1:5)");

testFail("\n\n\n{",
         "Unexpected token (4:1)");

testFail("\n/* Some multiline\ncomment */\n)",
         "Unexpected token (4:0)");

testFail("{ set 1 }",
         "Unexpected token (1:6)");

testFail("{ get 2 }",
         "Unexpected token (1:6)");

testFail("({ set: s(if) { } })",
         "Unexpected token (1:10)");

testFail("({ set s(.) { } })",
         "Unexpected token (1:9)");

testFail("({ set: s() { } })",
         "Unexpected token (1:12)");

testFail("({ set: s(a, b) { } })",
         "Unexpected token (1:16)");

testFail("({ get: g(d) { } })",
         "Unexpected token (1:13)");

testFail("({ get i() { }, i: 42 })",
         "Redefinition of property (1:16)");

testFail("({ i: 42, get i() { } })",
         "Redefinition of property (1:14)");

testFail("({ set i(x) { }, i: 42 })",
         "Redefinition of property (1:17)");

testFail("({ i: 42, set i(x) { } })",
         "Redefinition of property (1:14)");

testFail("({ get i() { }, get i() { } })",
         "Redefinition of property (1:20)");

testFail("({ set i(x) { }, set i(x) { } })",
         "Redefinition of property (1:21)");

testFail("function t(...) { }",
         "Unexpected token (1:11)");

testFail("function t(...) { }",
         "Unexpected token (1:14)",
         { ecmaVersion: 6 });

testFail("function t(...rest, b) { }",
         "Unexpected token (1:18)",
         { ecmaVersion: 6 });

testFail("function t(if) { }",
         "Unexpected token (1:11)");

testFail("function t(true) { }",
         "Unexpected token (1:11)");

testFail("function t(false) { }",
         "Unexpected token (1:11)");

testFail("function t(null) { }",
         "Unexpected token (1:11)");

testFail("function null() { }",
         "Unexpected token (1:9)");

testFail("function true() { }",
         "Unexpected token (1:9)");

testFail("function false() { }",
         "Unexpected token (1:9)");

testFail("function if() { }",
         "Unexpected token (1:9)");

testFail("a b;",
         "Unexpected token (1:2)");

testFail("if.a;",
         "Unexpected token (1:2)");

testFail("a if;",
         "Unexpected token (1:2)");

testFail("a class;",
         "Unexpected token (1:2)");

testFail("break\n",
         "Unsyntactic break (1:0)");

testFail("break 1;",
         "Unexpected token (1:6)");

testFail("continue\n",
         "Unsyntactic continue (1:0)");

testFail("continue 2;",
         "Unexpected token (1:9)");

testFail("throw",
         "Unexpected token (1:5)");

testFail("throw;",
         "Unexpected token (1:5)");

testFail("for (var i, i2 in {});",
         "Unexpected token (1:15)");

testFail("for ((i in {}));",
         "Unexpected token (1:14)");

testFail("for (i + 1 in {});",
         "Assigning to rvalue (1:5)");

testFail("for (+i in {});",
         "Assigning to rvalue (1:5)");

testFail("if(false)",
         "Unexpected token (1:9)");

testFail("if(false) doThis(); else",
         "Unexpected token (1:24)");

testFail("do",
         "Unexpected token (1:2)");

testFail("while(false)",
         "Unexpected token (1:12)");

testFail("for(;;)",
         "Unexpected token (1:7)");

testFail("with(x)",
         "Unexpected token (1:7)");

testFail("try { }",
         "Missing catch or finally clause (1:0)");

testFail("‿ = 10",
         "Unexpected character '‿' (1:0)");

testFail("if(true) let a = 1;",
         "Unexpected token (1:13)");

testFail("switch (c) { default: default: }",
         "Multiple default clauses (1:22)");

testFail("new X().\"s\"",
         "Unexpected token (1:8)");

testFail("/*",
         "Unterminated comment (1:0)");

testFail("/*\n\n\n",
         "Unterminated comment (1:0)");

testFail("/**",
         "Unterminated comment (1:0)");

testFail("/*\n\n*",
         "Unterminated comment (1:0)");

testFail("/*hello",
         "Unterminated comment (1:0)");

testFail("/*hello  *",
         "Unterminated comment (1:0)");

testFail("\n]",
         "Unexpected token (2:0)");

testFail("\r]",
         "Unexpected token (2:0)");

testFail("\r\n]",
         "Unexpected token (2:0)");

testFail("\n\r]",
         "Unexpected token (3:0)");

testFail("//\r\n]",
         "Unexpected token (2:0)");

testFail("//\n\r]",
         "Unexpected token (3:0)");

testFail("/a\\\n/",
         "Unterminated regular expression (1:1)");

testFail("//\r \n]",
         "Unexpected token (3:0)");

testFail("/*\r\n*/]",
         "Unexpected token (2:2)");

testFail("/*\n\r*/]",
         "Unexpected token (3:2)");

testFail("/*\r \n*/]",
         "Unexpected token (3:2)");

testFail("\\\\",
         "Expecting Unicode escape sequence \\uXXXX (1:1)");

testFail("\\u005c",
         "Invalid Unicode escape (1:2)");

testFail("\\x",
         "Expecting Unicode escape sequence \\uXXXX (1:1)");

testFail("\\u0000",
         "Invalid Unicode escape (1:2)");

testFail("‌ = []",
         "Unexpected character '‌' (1:0)");

testFail("‍ = []",
         "Unexpected character '‍' (1:0)");

testFail("\"\\",
         "Unterminated string constant (1:0)");

testFail("\"\\u",
         "Bad character escape sequence (1:0)");

testFail("return",
         "'return' outside of function (1:0)");

testFail("break",
         "Unsyntactic break (1:0)");

testFail("continue",
         "Unsyntactic continue (1:0)");

testFail("switch (x) { default: continue; }",
         "Unsyntactic continue (1:22)");

testFail("do { x } *",
         "Unexpected token (1:9)");

testFail("while (true) { break x; }",
         "Unsyntactic break (1:15)");

testFail("while (true) { continue x; }",
         "Unsyntactic continue (1:15)");

testFail("x: while (true) { (function () { break x; }); }",
         "Unsyntactic break (1:33)");

testFail("x: while (true) { (function () { continue x; }); }",
         "Unsyntactic continue (1:33)");

testFail("x: while (true) { (function () { break; }); }",
         "Unsyntactic break (1:33)");

testFail("x: while (true) { (function () { continue; }); }",
         "Unsyntactic continue (1:33)");

testFail("x: while (true) { x: while (true) { } }",
         "Label 'x' is already declared (1:18)");

testFail("(function () { 'use strict'; delete i; }())",
         "Deleting local variable in strict mode (1:29)");

testFail("(function () { 'use strict'; with (i); }())",
         "'with' in strict mode (1:29)");

testFail("function hello() {'use strict'; ({ i: 42, i: 42 }) }",
         "Redefinition of property (1:42)");

testFail("function hello() {'use strict'; ({ hasOwnProperty: 42, hasOwnProperty: 42 }) }",
         "Redefinition of property (1:55)");

testFail("function hello() {'use strict'; var eval = 10; }",
         "Binding eval in strict mode (1:36)");

testFail("function hello() {'use strict'; var arguments = 10; }",
         "Binding arguments in strict mode (1:36)");

testFail("function hello() {'use strict'; try { } catch (eval) { } }",
         "Binding eval in strict mode (1:47)");

testFail("function hello() {'use strict'; try { } catch (arguments) { } }",
         "Binding arguments in strict mode (1:47)");

testFail("function hello() {'use strict'; eval = 10; }",
         "Assigning to eval in strict mode (1:32)");

testFail("function hello() {'use strict'; arguments = 10; }",
         "Assigning to arguments in strict mode (1:32)");

testFail("function hello() {'use strict'; ++eval; }",
         "Assigning to eval in strict mode (1:34)");

testFail("function hello() {'use strict'; --eval; }",
         "Assigning to eval in strict mode (1:34)");

testFail("function hello() {'use strict'; ++arguments; }",
         "Assigning to arguments in strict mode (1:34)");

testFail("function hello() {'use strict'; --arguments; }",
         "Assigning to arguments in strict mode (1:34)");

testFail("function hello() {'use strict'; eval++; }",
         "Assigning to eval in strict mode (1:32)");

testFail("function hello() {'use strict'; eval--; }",
         "Assigning to eval in strict mode (1:32)");

testFail("function hello() {'use strict'; arguments++; }",
         "Assigning to arguments in strict mode (1:32)");

testFail("function hello() {'use strict'; arguments--; }",
         "Assigning to arguments in strict mode (1:32)");

testFail("function hello() {'use strict'; function eval() { } }",
         "Defining 'eval' in strict mode (1:41)");

testFail("function hello() {'use strict'; function arguments() { } }",
         "Defining 'arguments' in strict mode (1:41)");

testFail("function eval() {'use strict'; }",
         "Defining 'eval' in strict mode (1:9)");

testFail("function arguments() {'use strict'; }",
         "Defining 'arguments' in strict mode (1:9)");

testFail("function hello() {'use strict'; (function eval() { }()) }",
         "Defining 'eval' in strict mode (1:42)");

testFail("function hello() {'use strict'; (function arguments() { }()) }",
         "Defining 'arguments' in strict mode (1:42)");

testFail("(function eval() {'use strict'; })()",
         "Defining 'eval' in strict mode (1:10)");

testFail("(function arguments() {'use strict'; })()",
         "Defining 'arguments' in strict mode (1:10)");

testFail("function hello() {'use strict'; ({ s: function eval() { } }); }",
         "Defining 'eval' in strict mode (1:47)");

testFail("(function package() {'use strict'; })()",
         "Defining 'package' in strict mode (1:10)");

testFail("function hello() {'use strict'; ({ i: 10, set s(eval) { } }); }",
         "Defining 'eval' in strict mode (1:48)");

testFail("function hello() {'use strict'; ({ set s(eval) { } }); }",
         "Defining 'eval' in strict mode (1:41)");

testFail("function hello() {'use strict'; ({ s: function s(eval) { } }); }",
         "Defining 'eval' in strict mode (1:49)");

testFail("function hello(eval) {'use strict';}",
         "Defining 'eval' in strict mode (1:15)");

testFail("function hello(arguments) {'use strict';}",
         "Defining 'arguments' in strict mode (1:15)");

testFail("function hello() { 'use strict'; function inner(eval) {} }",
         "Defining 'eval' in strict mode (1:48)");

testFail("function hello() { 'use strict'; function inner(arguments) {} }",
         "Defining 'arguments' in strict mode (1:48)");

testFail("function hello() { 'use strict'; \"\\1\"; }",
         "Octal literal in strict mode (1:34)");

testFail("function hello() { 'use strict'; 021; }",
         "Invalid number (1:33)");

testFail("function hello() { 'use strict'; ({ \"\\1\": 42 }); }",
         "Octal literal in strict mode (1:37)");

testFail("function hello() { 'use strict'; ({ 021: 42 }); }",
         "Invalid number (1:36)");

testFail("function hello() { \"use strict\"; function inner() { \"octal directive\\1\"; } }",
         "Octal literal in strict mode (1:68)");

testFail("function hello() { \"use strict\"; var implements; }",
         "The keyword 'implements' is reserved (1:37)");

testFail("function hello() { \"use strict\"; var interface; }",
         "The keyword 'interface' is reserved (1:37)");

testFail("function hello() { \"use strict\"; var package; }",
         "The keyword 'package' is reserved (1:37)");

testFail("function hello() { \"use strict\"; var private; }",
         "The keyword 'private' is reserved (1:37)");

testFail("function hello() { \"use strict\"; var protected; }",
         "The keyword 'protected' is reserved (1:37)");

testFail("function hello() { \"use strict\"; var public; }",
         "The keyword 'public' is reserved (1:37)");

testFail("function hello() { \"use strict\"; var static; }",
         "The keyword 'static' is reserved (1:37)");

testFail("function hello(static) { \"use strict\"; }",
         "Defining 'static' in strict mode (1:15)");

testFail("function static() { \"use strict\"; }",
         "Defining 'static' in strict mode (1:9)");

testFail("\"use strict\"; function static() { }",
         "The keyword 'static' is reserved (1:23)");

testFail("function a(t, t) { \"use strict\"; }",
         "Argument name clash in strict mode (1:14)");

testFail("function a(eval) { \"use strict\"; }",
         "Defining 'eval' in strict mode (1:11)");

testFail("function a(package) { \"use strict\"; }",
         "Defining 'package' in strict mode (1:11)");

testFail("function a() { \"use strict\"; function b(t, t) { }; }",
         "Argument name clash in strict mode (1:43)");

testFail("(function a(t, t) { \"use strict\"; })",
         "Argument name clash in strict mode (1:15)");

testFail("function a() { \"use strict\"; (function b(t, t) { }); }",
         "Argument name clash in strict mode (1:44)");

testFail("(function a(eval) { \"use strict\"; })",
         "Defining 'eval' in strict mode (1:12)");

testFail("(function a(package) { \"use strict\"; })",
         "Defining 'package' in strict mode (1:12)");

testFail("\"use strict\";function foo(){\"use strict\";}function bar(){var v = 015}",
         "Invalid number (1:65)");

testFail("var this = 10;", "Unexpected token (1:4)");

testFail("throw\n10;", "Illegal newline after throw (1:5)");


// ECMA < 6 mode should work as before

testFail("const a;", "Unexpected token (1:6)");

testFail("let x;", "Unexpected token (1:4)");

testFail("const a = 1;", "Unexpected token (1:6)");

testFail("let a = 1;", "Unexpected token (1:4)");

testFail("for(const x = 0;;);", "Unexpected token (1:10)");

testFail("for(let x = 0;;);", "Unexpected token (1:8)");

test("let++", {
  type: "Program",
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  },
  body: [
    {
      type: "ExpressionStatement",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      },
      expression: {
        type: "UpdateExpression",
        loc: {
          start: {
            line: 1,
            column: 0
          },
          end: {
            line: 1,
            column: 5
          }
        },
        operator: "++",
        prefix: false,
        argument: {
          type: "Identifier",
          loc: {
            start: {
              line: 1,
              column: 0
            },
            end: {
              line: 1,
              column: 3
            }
          },
          name: "let"
        }
      }
    }
  ]
});

// ECMA 6 support

test("let x", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        }
      ],
      kind: "let",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 5
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 5
    }
  }
}, {ecmaVersion: 6, locations: true});

test("let x, y;", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 5
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 7
              },
              end: {
                line: 1,
                column: 8
              }
            }
          },
          init: null,
          loc: {
            start: {
              line: 1,
              column: 7
            },
            end: {
              line: 1,
              column: 8
            }
          }
        }
      ],
      kind: "let",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 9
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 9
    }
  }
}, {ecmaVersion: 6, locations: true});

test("let x = 42", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        }
      ],
      kind: "let",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 10
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 10
    }
  }
}, {ecmaVersion: 6, locations: true});

test("let eval = 42, arguments = 42", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "eval",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 8
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 11
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 13
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "arguments",
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 24
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 27
              },
              end: {
                line: 1,
                column: 29
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 15
            },
            end: {
              line: 1,
              column: 29
            }
          }
        }
      ],
      kind: "let",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
}, {ecmaVersion: 6, locations: true});

test("let x = 14, y = 3, z = 1977", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 4
              },
              end: {
                line: 1,
                column: 5
              }
            }
          },
          init: {
            type: "Literal",
            value: 14,
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 4
            },
            end: {
              line: 1,
              column: 10
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 12
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          init: {
            type: "Literal",
            value: 3,
            loc: {
              start: {
                line: 1,
                column: 16
              },
              end: {
                line: 1,
                column: 17
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 12
            },
            end: {
              line: 1,
              column: 17
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 19
              },
              end: {
                line: 1,
                column: 20
              }
            }
          },
          init: {
            type: "Literal",
            value: 1977,
            loc: {
              start: {
                line: 1,
                column: 23
              },
              end: {
                line: 1,
                column: 27
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 19
            },
            end: {
              line: 1,
              column: 27
            }
          }
        }
      ],
      kind: "let",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 27
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 27
    }
  }
}, {ecmaVersion: 6, locations: true});

test("for(let x = 0;;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 8
                },
                end: {
                  line: 1,
                  column: 9
                }
              }
            },
            init: {
              type: "Literal",
              value: 0,
              loc: {
                start: {
                  line: 1,
                  column: 12
                },
                end: {
                  line: 1,
                  column: 13
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          }
        ],
        kind: "let",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 13
          }
        }
      },
      test: null,
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 16
          },
          end: {
            line: 1,
            column: 17
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 17
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 17
    }
  }
}, {ecmaVersion: 6, locations: true});

test("for(let x = 0, y = 1;;);", {
  type: "Program",
  body: [
    {
      type: "ForStatement",
      init: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 8
                },
                end: {
                  line: 1,
                  column: 9
                }
              }
            },
            init: {
              type: "Literal",
              value: 0,
              loc: {
                start: {
                  line: 1,
                  column: 12
                },
                end: {
                  line: 1,
                  column: 13
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 8
              },
              end: {
                line: 1,
                column: 13
              }
            }
          },
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "y",
              loc: {
                start: {
                  line: 1,
                  column: 15
                },
                end: {
                  line: 1,
                  column: 16
                }
              }
            },
            init: {
              type: "Literal",
              value: 1,
              loc: {
                start: {
                  line: 1,
                  column: 19
                },
                end: {
                  line: 1,
                  column: 20
                }
              }
            },
            loc: {
              start: {
                line: 1,
                column: 15
              },
              end: {
                line: 1,
                column: 20
              }
            }
          }
        ],
        kind: "let",
        loc: {
          start: {
            line: 1,
            column: 4
          },
          end: {
            line: 1,
            column: 20
          }
        }
      },
      test: null,
      update: null,
      body: {
        type: "EmptyStatement",
        loc: {
          start: {
            line: 1,
            column: 23
          },
          end: {
            line: 1,
            column: 24
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 24
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 24
    }
  }
}, {ecmaVersion: 6, locations: true});

test("for (let x in list) process(x);", {
  type: "Program",
  body: [
    {
      type: "ForInStatement",
      left: {
        type: "VariableDeclaration",
        declarations: [
          {
            type: "VariableDeclarator",
            id: {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 9
                },
                end: {
                  line: 1,
                  column: 10
                }
              }
            },
            init: null,
            loc: {
              start: {
                line: 1,
                column: 9
              },
              end: {
                line: 1,
                column: 10
              }
            }
          }
        ],
        kind: "let",
        loc: {
          start: {
            line: 1,
            column: 5
          },
          end: {
            line: 1,
            column: 10
          }
        }
      },
      right: {
        type: "Identifier",
        name: "list",
        loc: {
          start: {
            line: 1,
            column: 14
          },
          end: {
            line: 1,
            column: 18
          }
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
              start: {
                line: 1,
                column: 20
              },
              end: {
                line: 1,
                column: 27
              }
            }
          },
          arguments: [
            {
              type: "Identifier",
              name: "x",
              loc: {
                start: {
                  line: 1,
                  column: 28
                },
                end: {
                  line: 1,
                  column: 29
                }
              }
            }
          ],
          loc: {
            start: {
              line: 1,
              column: 20
            },
            end: {
              line: 1,
              column: 30
            }
          }
        },
        loc: {
          start: {
            line: 1,
            column: 20
          },
          end: {
            line: 1,
            column: 31
          }
        }
      },
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 31
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 31
    }
  }
}, {ecmaVersion: 6, locations: true});

test("const x = 42", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 6
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 10
              },
              end: {
                line: 1,
                column: 12
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 12
            }
          }
        }
      ],
      kind: "const",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 12
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 12
    }
  }
}, {ecmaVersion: 6, locations: true});

test("const eval = 42, arguments = 42", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "eval",
            loc: {
              start: {
                line: 1,
                column: 6
              },
              end: {
                line: 1,
                column: 10
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 13
              },
              end: {
                line: 1,
                column: 15
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 15
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "arguments",
            loc: {
              start: {
                line: 1,
                column: 17
              },
              end: {
                line: 1,
                column: 26
              }
            }
          },
          init: {
            type: "Literal",
            value: 42,
            loc: {
              start: {
                line: 1,
                column: 29
              },
              end: {
                line: 1,
                column: 31
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 17
            },
            end: {
              line: 1,
              column: 31
            }
          }
        }
      ],
      kind: "const",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 31
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 31
    }
  }
}, {ecmaVersion: 6, locations: true});

test("const x = 14, y = 3, z = 1977", {
  type: "Program",
  body: [
    {
      type: "VariableDeclaration",
      declarations: [
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "x",
            loc: {
              start: {
                line: 1,
                column: 6
              },
              end: {
                line: 1,
                column: 7
              }
            }
          },
          init: {
            type: "Literal",
            value: 14,
            loc: {
              start: {
                line: 1,
                column: 10
              },
              end: {
                line: 1,
                column: 12
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 6
            },
            end: {
              line: 1,
              column: 12
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "y",
            loc: {
              start: {
                line: 1,
                column: 14
              },
              end: {
                line: 1,
                column: 15
              }
            }
          },
          init: {
            type: "Literal",
            value: 3,
            loc: {
              start: {
                line: 1,
                column: 18
              },
              end: {
                line: 1,
                column: 19
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 14
            },
            end: {
              line: 1,
              column: 19
            }
          }
        },
        {
          type: "VariableDeclarator",
          id: {
            type: "Identifier",
            name: "z",
            loc: {
              start: {
                line: 1,
                column: 21
              },
              end: {
                line: 1,
                column: 22
              }
            }
          },
          init: {
            type: "Literal",
            value: 1977,
            loc: {
              start: {
                line: 1,
                column: 25
              },
              end: {
                line: 1,
                column: 29
              }
            }
          },
          loc: {
            start: {
              line: 1,
              column: 21
            },
            end: {
              line: 1,
              column: 29
            }
          }
        }
      ],
      kind: "const",
      loc: {
        start: {
          line: 1,
          column: 0
        },
        end: {
          line: 1,
          column: 29
        }
      }
    }
  ],
  loc: {
    start: {
      line: 1,
      column: 0
    },
    end: {
      line: 1,
      column: 29
    }
  }
}, {ecmaVersion: 6, locations: true});

testFail("const a;", "Unexpected token (1:7)", {ecmaVersion: 6});

testFail("for(const x = 0;;);", "Unexpected token (1:4)", {ecmaVersion: 6});

testFail("for(x of a);", "Unexpected token (1:6)");

testFail("for(var x of a);", "Unexpected token (1:10)");

// Assertion Tests
test(function TestComments() {
    // Bear class
    function Bear(x,y,z) {
      this.position = [x||0,y||0,z||0]
    }

    Bear.prototype.roar = function(message) {
      return 'RAWWW: ' + message; // Whatever
    };

    function Cat() {
    /* 1
       2
       3*/
    }

    Cat.prototype.roar = function(message) {
      return 'MEOOWW: ' + /*stuff*/ message;
    };
}.toString().replace(/\r\n/g, '\n'), {}, {
  onComment: [
    {type: "Line", value: " Bear class"},
    {type: "Line", value: " Whatever"},
    {type: "Block",  value: [
            " 1",
      "       2",
      "       3"
    ].join('\n')},
    {type: "Block", value: "stuff"}
  ]
});

test("<!--\n;", {
  type: "Program",
  body: [{
    type: "EmptyStatement"
  }]
});

test("\nfunction plop() {\n'use strict';\n/* Comment */\n}", {}, {
  locations: true,
  onComment: [{
    type: "Block",
    value: " Comment ",
    loc: {
      start: { line: 4, column: 0 },
      end: { line: 4, column: 13 }
    }
  }]
});

test("// line comment", {}, {
  locations: true,
  onComment: [{
    type: "Line",
    value: " line comment",
    loc: {
      start: { line: 1, column: 0 },
      end: { line: 1, column: 15 }
    }
  }]
});

test("<!-- HTML comment", {}, {
  locations: true,
  onComment: [{
    type: "Line",
    value: " HTML comment",
    loc: {
      start: { line: 1, column: 0 },
      end: { line: 1, column: 17 }
    }
  }]
});

test(";\n--> HTML comment", {}, {
  locations: true,
  onComment: [{
    type: "Line",
    value: " HTML comment",
    loc: {
      start: { line: 2, column: 0 },
      end: { line: 2, column: 16 }
    }
  }]
});

var tokTypes = acorn.tokTypes;

test('var x = (1 + 2)', {}, {
  locations: true,
  onToken: [
    {
      type: tokTypes._var,
      value: "var",
      loc: {
        start: {line: 1, column: 0},
        end: {line: 1, column: 3}
      }
    },
    {
      type: tokTypes.name,
      value: "x",
      loc: {
        start: {line: 1, column: 4},
        end: {line: 1, column: 5}
      }
    },
    {
      type: tokTypes.eq,
      value: "=",
      loc: {
        start: {line: 1, column: 6},
        end: {line: 1, column: 7}
      }
    },
    {
      type: tokTypes.parenL,
      value: undefined,
      loc: {
        start: {line: 1, column: 8},
        end: {line: 1, column: 9}
      }
    },
    {
      type: tokTypes.num,
      value: 1,
      loc: {
        start: {line: 1, column: 9},
        end: {line: 1, column: 10}
      }
    },
    {
      type: {binop: 9, prefix: true, beforeExpr: true},
      value: "+",
      loc: {
        start: {line: 1, column: 11},
        end: {line: 1, column: 12}
      }
    },
    {
      type: tokTypes.num,
      value: 2,
      loc: {
        start: {line: 1, column: 13},
        end: {line: 1, column: 14}
      }
    },
    {
      type: tokTypes.parenR,
      value: undefined,
      loc: {
        start: {line: 1, column: 14},
        end: {line: 1, column: 15}
      }
    },
    {
      type: tokTypes.eof,
      value: undefined,
      loc: {
        start: {line: 1, column: 15},
        end: {line: 1, column: 15}
      }
    }
  ]
});

test("function f(f) { 'use strict'; }", {});
