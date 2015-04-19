require(__dirname).test({
  xml: '<r>&rfloor; ' +
       '&spades; &copy; &rarr; &amp; ' +
        '&lt; < <  <   < &gt; &real; &weierp; &euro;</r>',
  expect: [
    ['opentag', {'name':'R', attributes:{}, isSelfClosing: false}],
    ['text', '⌋ ♠ © → & < < <  <   < > ℜ ℘ €'],
    ['closetag', 'R']
  ]
});
