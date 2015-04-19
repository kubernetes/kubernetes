require(__dirname).test({
  xml: '<Р>тест</Р>',
  expect: [
    ['opentag', {'name':'Р', attributes:{}, isSelfClosing: false}],
    ['text', 'тест'],
    ['closetag', 'Р']
  ]
});
