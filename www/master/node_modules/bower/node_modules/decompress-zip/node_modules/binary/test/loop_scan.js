var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('loop scan', function (t) {
    t.plan(8 + 6 + 2);
    
    var em = new EventEmitter;
    
    binary.stream(em)
        .loop(function (end) {
            var vars_ = this.vars;
            this
                .scan('filler', 'BEGINMSG')
                .buffer('cmd', 3)
                .word8('num')
                .tap(function (vars) {
                    t.strictEqual(vars, vars_);
                    if (vars.num != 0x02 && vars.num != 0x06) {
                        t.same(vars.filler.length, 0);
                    }
                    if (vars.cmd.toString() == 'end') end();
                })
            ;
        })
        .tap(function (vars) {
            t.same(vars.cmd.toString(), 'end');
			t.same(vars.num, 0x08);
        })
    ;
    
    setTimeout(function () {
        em.emit('data', new Buffer(
            'BEGINMSGcmd\x01'
            + 'GARBAGEDATAXXXX'
            + 'BEGINMSGcmd\x02'
            + 'BEGINMSGcmd\x03'
        ));
    }, 10);
    
    setTimeout(function () {
        em.emit('data', new Buffer(
            'BEGINMSGcmd\x04'
            + 'BEGINMSGcmd\x05'
            + 'GARBAGEDATAXXXX'
            + 'BEGINMSGcmd\x06'
        ));
        em.emit('data', new Buffer('BEGINMSGcmd\x07'));
    }, 20);
    
    setTimeout(function () {
        em.emit('data', new Buffer('BEGINMSGend\x08'));
    }, 30);
});
