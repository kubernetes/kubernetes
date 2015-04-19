var colors = require('ansicolors');

// Change the below definitions in order to tweak the color theme.
module.exports = {

    'Boolean': {
      'true'   :  undefined
    , 'false'  :  undefined
    , _default :  colors.yellow
    }

  , 'Identifier': {
      'undefined' :  colors.yellow
    , 'self'      :  colors.yellow
    , 'type'      :  colors.yellow
    , 'value'     :  colors.yellow
    , 'console'   :  undefined
    , 'log'       :  colors.blue
    , 'warn'      :  colors.blue
    , 'error'     :  colors.blue
    , 'join'      :  colors.blue
    , _default    :  function (s, info) {
        var prevToken = info.tokens[info.tokenIndex - 1];
        var nextToken = info.tokens[info.tokenIndex + 1];

        return (nextToken
            && nextToken.type === 'Punctuator'
            && nextToken.value === '('
            && prevToken
            && prevToken.type === 'Keyword'
            && prevToken.value === 'function'
          ) ? colors.blue(s) : colors.white(s);
      }
    }

  , 'Null': {
      _default: colors.yellow
    }

  , 'Numeric': {
      _default: colors.yellow
    }

  , 'String': {
      _default: function (s, info) {
        var nextToken = info.tokens[info.tokenIndex + 1];

        // show keys of object literals and json in different color
        return (nextToken && nextToken.type === 'Punctuator' && nextToken.value === ':') 
          ? colors.green(s)
          : colors.brightGreen(s);
      }
    }

  , 'Keyword': {
      'break'       :  colors.magenta

    , 'case'        :  colors.magenta
    , 'catch'       :  colors.magenta
    , 'continue'    :  colors.magenta

    , 'debugger'    :  colors.magenta
    , 'default'     :  colors.magenta
    , 'delete'      :  colors.red
    , 'do'          :  colors.magenta

    , 'else'        :  colors.magenta

    , 'finally'     :  colors.magenta
    , 'for'         :  colors.magenta
    , 'function'    :  colors.magenta

    , 'if'          :  colors.magenta
    , 'in'          :  colors.cyan
    , 'instanceof'  :  colors.cyan

    , 'new'         :  colors.cyan
    , 'return'      :  colors.magenta
    , 'switch'      :  colors.magenta

    , 'this'        :  colors.red
    , 'throw'       :  colors.magenta
    , 'try'         :  colors.magenta
    , 'typeof'      :  colors.cyan

    , 'var'         :  colors.magenta
    , 'void'        :  colors.magenta

    , 'while'       :  colors.magenta
    , 'with'        :  colors.cyan
    , _default      :  colors.white
  }
  , 'Punctuator': {
      ';': colors.white
    , '.': colors.white
    , ',': colors.white

    , '{': colors.white
    , '}': colors.white
    , '(': colors.white
    , ')': colors.white
    , '[': colors.white
    , ']': colors.white

    , '<': undefined
    , '>': undefined
    , '+': undefined
    , '-': undefined
    , '*': undefined
    , '%': undefined
    , '&': undefined
    , '|': colors.white
    , '^': undefined
    , '!': undefined
    , '~': undefined
    , '?': colors.white
    , ':': colors.white
    , '=': undefined

    , '<=': undefined
    , '>=': undefined
    , '==': undefined
    , '!=': undefined
    , '++': undefined
    , '--': undefined
    , '<<': undefined
    , '>>': undefined
    , '&&': undefined
    , '||': undefined
    , '+=': undefined
    , '-=': undefined
    , '*=': undefined
    , '%=': undefined
    , '&=': undefined
    , '|=': undefined
    , '^=': undefined
    , '/=': undefined

    , '===': undefined
    , '!==': undefined
    , '>>>': undefined
    , '<<=': undefined
    , '>>=': undefined
    
    , '>>>=': undefined

    , _default: colors.cyan
  }

    // line comment
  , Line: {
     _default: colors.brightBlack
    }

    /* block comment */
  , Block: {
     _default: colors.brightBlack
    }

  , _default: undefined
};
