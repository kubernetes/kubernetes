// set this really low so that I don't have to put 64 MB of xml in here.
var sax = require("../lib/sax")
var bl = sax.MAX_BUFFER_LENGTH
sax.MAX_BUFFER_LENGTH = 5;

require(__dirname).test({
  expect : [
    ["error", "Max buffer length exceeded: tagName\nLine: 0\nColumn: 15\nChar: "],
    ["error", "Max buffer length exceeded: tagName\nLine: 0\nColumn: 30\nChar: "],
    ["error", "Max buffer length exceeded: tagName\nLine: 0\nColumn: 45\nChar: "],
    ["opentag", {
     "name": "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ",
     "attributes": {},
     "isSelfClosing": false
    }],
    ["text", "yo"],
    ["closetag", "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ"]
  ]
}).write("<abcdefghijklmn")
  .write("opqrstuvwxyzABC")
  .write("DEFGHIJKLMNOPQR")
  .write("STUVWXYZ>")
  .write("yo")
  .write("</abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>")
  .close();
sax.MAX_BUFFER_LENGTH = bl
