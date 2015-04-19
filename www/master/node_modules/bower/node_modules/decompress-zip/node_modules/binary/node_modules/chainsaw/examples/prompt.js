var Chainsaw = require('chainsaw');
var Lazy = require('lazy');

module.exports = Prompt;
function Prompt (stream) {
    var waiting = [];
    var lines = [];
    var lazy = Lazy(stream).lines.map(String)
        .forEach(function (line) {
            if (waiting.length) {
                var w = waiting.shift();
                w(line);
            }
            else lines.push(line);
        })
    ;
    
    var vars = {};
    return Chainsaw(function (saw) {
        this.getline = function (f) {
            var g = function (line) {
                saw.nest(f, line, vars);
            };
            
            if (lines.length) g(lines.shift());
            else waiting.push(g);
        };
        
        this.do = function (cb) {
            saw.nest(cb, vars);
        };
    });
}

var util = require('util');
if (__filename === process.argv[1]) {
    var stdin = process.openStdin();
    Prompt(stdin)
        .do(function () {
            util.print('x = ');
        })
        .getline(function (line, vars) {
            vars.x = parseInt(line, 10);
        })
        .do(function () {
            util.print('y = ');
        })
        .getline(function (line, vars) {
            vars.y = parseInt(line, 10);
        })
        .do(function (vars) {
            if (vars.x + vars.y < 10) {
                util.print('z = ');
                this.getline(function (line) {
                    vars.z = parseInt(line, 10);
                })
            }
            else {
                vars.z = 0;
            }
        })
        .do(function (vars) {
            console.log('x + y + z = ' + (vars.x + vars.y + vars.z));
            process.exit();
        })
    ;
}
