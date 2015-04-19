var colors = require('ansicolors');

// Change the below definitions in order to tweak the color theme.
module.exports = {

    'Boolean': {
      // changed from default
      'true'   :  colors.red

    , 'false'  :  undefined
    , _default :  colors.brightRed
    }

  , 'Identifier': {
      'undefined' :  colors.brightBlack
    , 'self'      :  colors.brightRed
    , 'console'   :  colors.blue
    , 'log'       :  colors.blue
    , 'warn'      :  colors.red
    , 'error'     :  colors.brightRed
    //
      // changed from default
    , _default    :  colors.brightCyan
    }

  , 'Null': {
      _default: colors.brightBlack
    }

  , 'Numeric': {
      _default: colors.blue
    }

  , 'String': {
      _default: colors.brightGreen
    }

  , 'Keyword': {
      'break'       :  undefined

    , 'case'        :  undefined
    , 'catch'       :  colors.cyan
    , 'continue'    :  undefined

    , 'debugger'    :  undefined
    , 'default'     :  undefined
    , 'delete'      :  colors.red
    , 'do'          :  undefined

    , 'else'        :  undefined

    , 'finally'     :  colors.cyan
    , 'for'         :  undefined
    , 'function'    :  undefined

    , 'if'          :  undefined
    , 'in'          :  undefined
    , 'instanceof'  :  undefined

    , 'new'         :  colors.red
    , 'return'      :  colors.red
    , 'switch'      :  undefined

    , 'this'        :  colors.brightRed
    , 'throw'       :  undefined
    , 'try'         :  colors.cyan
    , 'typeof'      :  undefined

    , 'var'         :  colors.green
    , 'void'        :  undefined

    , 'while'       :  undefined
    , 'with'        :  undefined
    , _default      :  colors.brightBlue
  }
  , 'Punctuator': {
      ';': colors.brightBlack
    , '.': colors.green  
    , ',': colors.green  

    , '{': colors.yellow
    , '}': colors.yellow
    , '(': colors.brightBlack  
    , ')': colors.brightBlack  
    , '[': colors.yellow
    , ']': colors.yellow

    , '<': undefined
    , '>': undefined
    , '+': undefined
    , '-': undefined
    , '*': undefined
    , '%': undefined
    , '&': undefined
    , '|': undefined
    , '^': undefined
    , '!': undefined
    , '~': undefined
    , '?': undefined
    , ':': undefined
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

    , _default: colors.brightYellow
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
