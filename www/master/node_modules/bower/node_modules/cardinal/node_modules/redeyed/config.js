/* 
 * Copy this file and use it as a starting point for your redeyed config.
 * Just fill in the tokens you want to surround/replace.
 * Keep in mind that more specific configurations override less specific ones.
 */

module.exports = {

    'Boolean': {
      'true'   :  undefined
    , 'false'  :  undefined
    , _default :  undefined
    }

  , 'Identifier': {
      _default: undefined
    }

  , 'Null': {
      _default: undefined
    }

  , 'Numeric': {
      _default: undefined
    }

  , 'String': {
      _default: undefined
    }

  , 'Keyword': {
      'break'       :  undefined

    , 'case'        :  undefined
    , 'catch'       :  undefined
    , 'continue'    :  undefined

    , 'debugger'    :  undefined
    , 'default'     :  undefined
    , 'delete'      :  undefined
    , 'do'          :  undefined

    , 'else'        :  undefined

    , 'finally'     :  undefined
    , 'for'         :  undefined
    , 'function'    :  undefined

    , 'if'          :  undefined
    , 'in'          :  undefined
    , 'instanceof'  :  undefined

    , 'new'         :  undefined
    , 'return'      :  undefined
    , 'switch'      :  undefined

    , 'this'        :  undefined
    , 'throw'       :  undefined
    , 'try'         :  undefined
    , 'typeof'      :  undefined

    , 'var'         :  undefined
    , 'void'        :  undefined

    , 'while'       :  undefined
    , 'with'        :  undefined
    , _default      :  undefined
  }
  , 'Punctuator': {
      ';': undefined  
    , '.': undefined  
    , ',': undefined  

    , '{': undefined  
    , '}': undefined  
    , '(': undefined  
    , ')': undefined  
    , '[': undefined
    , ']': undefined

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

    , _default: undefined
  }

  // line comment
  , Line: {
     _default: undefined
    }

  /* block comment */
  , Block: {
     _default: undefined
    }

  , _default: undefined
};
