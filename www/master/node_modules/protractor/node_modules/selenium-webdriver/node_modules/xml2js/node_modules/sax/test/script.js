require(__dirname).test({
  xml : "<html><head><script>if (1 < 0) { console.log('elo there'); }</script></head></html>",
  expect : [
    ["opentag", {"name": "HTML","attributes": {}, "isSelfClosing": false}],
    ["opentag", {"name": "HEAD","attributes": {}, "isSelfClosing": false}],
    ["opentag", {"name": "SCRIPT","attributes": {}, "isSelfClosing": false}],
    ["script", "if (1 < 0) { console.log('elo there'); }"],
    ["closetag", "SCRIPT"],
    ["closetag", "HEAD"],
    ["closetag", "HTML"]
  ]
});
