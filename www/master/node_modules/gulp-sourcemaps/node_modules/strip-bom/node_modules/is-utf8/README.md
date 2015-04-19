#utf8 detector

Detect if a Buffer is utf8 encoded

    var fs = require('fs');
    var isUtf8 = require('is-utf8');
    var ansi = fs.readFileSync('ansi.txt');
    var utf8 = fs.readFileSync('utf8.txt');
    
    console.log('ansi.txt is utf8: '+isUtf8(ansi));
    console.log('utf8.txt is utf8: '+isUtf8(utf8));
    
