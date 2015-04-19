var parser = require(__dirname).test({
  expect: [
    ['opentag', {'name':'T', attributes:{}, isSelfClosing: false}],
    ['text', 'flush'],
    ['text', 'rest'],
    ['closetag', 'T'],
  ]
});

parser.write('<T>flush');
parser.flush();
parser.write('rest</T>');
parser.close();
