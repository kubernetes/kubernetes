var path    =  require('path')
  , fs      =  require('fs')
  , redeyed =  require('..')
  , vm      =  require('vm')
  ;

var samplePath =  path.join(__dirname, 'sources', 'log.js')
  , origCode   =  fs.readFileSync(samplePath, 'utf-8')
  , kinds = ['silly', 'info', 'warn', 'error' ]
  ;

function replaceConsole(s, info) {
  var code        =  info.code
    , idx         =  info.tokenIndex
    , tokens      =  info.tokens
    , next        =  tokens[idx + 1].value
    , kind        =  tokens[idx + 2].value
    , openParen   =  tokens[idx + 3].value
    , firstArgTkn =  tokens[idx + 4]
    , argIdx      =  idx + 3
    , open
    , tkn
    ;

  if (kind === 'log') kind = 'silly';

  // not a console.xxx(...) statement? -> just return original
  if (next !== '.' || !~kinds.indexOf(kind) || openParen !== '(') return s;

  // skip past arguments to console.xxx all args from ( to )
  open = 1;
  while (open) {
    tkn = tokens[++argIdx];

    // count open parens vs. closed ones to handle things like console.log(new Error('..'));
    if (tkn.value === '(') open++;
    if (tkn.value === ')') open--;
  }

  // tkn now is the last closing paren
  var argsIncludingClosingParen =  code.slice(firstArgTkn.range[0], tkn.range[1])
    , result                    =  'log.' + kind + '("main-logger", ' + argsIncludingClosingParen;
  
  // tell redeyed to skip the entire console.xxx(..) statement since we are replacing it all
  return { replacement: result, skipPastToken: tkn }; 
}

function transformAndRun () {
  var config = {
        Identifier: { console: replaceConsole }
      }
    , code        =  redeyed(origCode, config).code
    , context     =  vm.createContext({ require: require });

  console.log('Original code:\n', origCode);
  console.log('\nlog calls replaced:\n', code);
  console.log('\nLets run it:');
  vm.runInContext(code, context, 'transformed-log.vm');
}

transformAndRun();
