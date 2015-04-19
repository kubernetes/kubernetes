window.sampleConfig = {

    'Boolean': {
      'true'   :  undefined
    , 'false'  :  undefined
    , _default :  '?:?'
    }

  , 'Identifier': {
      _default: '-> : <-'
    }

  , 'Null': {
      _default: '**:**'
    }

  , 'Numeric': {
      _default: 'n:N'
    }

  , 'String': {
      _default: 'string -> :'
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
    , _default      :  ': <- keyword'
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
