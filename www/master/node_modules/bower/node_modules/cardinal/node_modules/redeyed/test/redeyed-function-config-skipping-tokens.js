'use strict';
/*jshint asi: true*/

var test = require('tap').test
  , util = require('util')
  , redeyed = require('..')

function inspect (obj) {
  return util.inspect(obj, false, 5, true)
}

          
test('given i skip 2 more tokens after each semicolon', function (t) {
  var calls = 0
    , opts = {
        Punctuator: {
          ';':  function identity (s, info) {
                  // tell it to skip past second to last token that is 2 ahead of the current one
                  calls++
                  var skipToken = info.tokens[info.tokenIndex + 2]
                  return skipToken ? { replacement: s, skipPastToken: skipToken } : s;
                }
        }
      }
    ;

  [  { code: ';;;'                   ,  expectedCalls: 1 }
  ,  { code: ';;;;'                  ,  expectedCalls: 2 }
  ,  { code: '; ; ; ;'               ,  expectedCalls: 2 }
  ,  { code: ';;; ;;; ;;; ;'         ,  expectedCalls: 4 }
  ,  { code: ';;; ;;; ;;; ;;; ;'     ,  expectedCalls: 5 }
  ,  { code: ';;; ;;; ;;; ;;; ;;;'   ,  expectedCalls: 5 }
  ,  { code: ';;; ;;; ;;; ;;; ;;; ;' ,  expectedCalls: 6 }
  ].forEach(function (x) {
      calls = 0
      redeyed(x.code, opts);
      t.equals(calls, x.expectedCalls, 'calls ' + x.expectedCalls + ' times for ' + x.code)
    });
  t.end()
})

test('replace log', function (t) {
  var kinds = [ 'info', 'warn', 'error' ]
    , opts = {
        Identifier: { 
          console: function replaceLog(s, info) {
            var code        =  info.code
              , idx         =  info.tokenIndex
              , tokens      =  info.tokens
              , kind        =  tokens[idx + 2].value
              , openParen   =  tokens[idx + 3].value
              , firstArgTkn =  tokens[idx + 4]
              , argIdx      =  idx + 3
              , open
              , tkn
              ;

            open = 1;
            while (open) {
              tkn = tokens[++argIdx];

              if (tkn.value === '(') open++;
              if (tkn.value === ')') open--;
            }

            var argsIncludingClosingParen =  code.slice(firstArgTkn.range[0], tkn.range[1])
              , result                    =  'log.' + kind + '("main-logger", ' + argsIncludingClosingParen;
            
            return { replacement: result, skipPastToken: tkn }; 
          }
        }
      }

   , origCode = [
       'console.info("info ", 1);'
     , 'console.warn("warn ", 3);'
     , 'console.error("error ", new Error("oh my!"));'
    ].join('\n')
  
  , expectedCode = [
      'log.info("main-logger", "info ", 1));'
    , 'log.warn("main-logger", "warn ", 3));'
    , 'log.error("main-logger", "error ", new Error("oh my!")));'
    ].join('\n')
  , code = redeyed(origCode, opts).code

  t.equals(code, expectedCode, 'transforms all log statements')
  t.end()
});
