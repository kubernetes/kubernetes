var assert = require('assert');
var Chainsaw = require('../index');

exports.getset = function () {
    var to = setTimeout(function () {
        assert.fail('builder never fired');
    }, 1000);
    
    var ch = Chainsaw(function (saw) {
        clearTimeout(to);
        var num = 0;
        
        this.get = function (cb) {
            cb(num);
            saw.next();
        };
        
        this.set = function (n) {
            num = n;
            saw.next();
        };
        
        var ti = setTimeout(function () {
            assert.fail('end event not emitted');
        }, 50);
        
        saw.on('end', function () {
            clearTimeout(ti);
            assert.equal(times, 3);
        });
    });
    
    var times = 0;
    ch
        .get(function (x) {
            assert.equal(x, 0);
            times ++;
        })
        .set(10)
        .get(function (x) {
            assert.equal(x, 10);
            times ++;
        })
        .set(20)
        .get(function (x) {
            assert.equal(x, 20);
            times ++;
        })
    ;
};

exports.nest = function () {
    var ch = (function () {
        var vars = {};
        return Chainsaw(function (saw) {
            this.do = function (cb) {
                saw.nest(cb, vars);
            };
        });
    })();
    
    var order = [];
    var to = setTimeout(function () {
        assert.fail("Didn't get to the end");
    }, 50);
    
    ch
        .do(function (vars) {
            vars.x = 'y';
            order.push(1);
            
            this
                .do(function (vs) {
                    order.push(2);
                    vs.x = 'x';
                })
                .do(function (vs) {
                    order.push(3);
                    vs.z = 'z';
                })
            ;
        })
        .do(function (vars) {
            vars.y = 'y';
            order.push(4);
        })
        .do(function (vars) {
            assert.eql(order, [1,2,3,4]);
            assert.eql(vars, { x : 'x', y : 'y', z : 'z' });
            clearTimeout(to);
        })
    ;
};

exports.nestWait = function () {
    var ch = (function () {
        var vars = {};
        return Chainsaw(function (saw) {
            this.do = function (cb) {
                saw.nest(cb, vars);
            };
            
            this.wait = function (n) {
                setTimeout(function () {
                    saw.next();
                }, n);
            };
        });
    })();
    
    var order = [];
    var to = setTimeout(function () {
        assert.fail("Didn't get to the end");
    }, 1000);
    
    var times = {};
    
    ch
        .do(function (vars) {
            vars.x = 'y';
            order.push(1);
            
            this
                .do(function (vs) {
                    order.push(2);
                    vs.x = 'x';
                    times.x = Date.now();
                })
                .wait(50)
                .do(function (vs) {
                    order.push(3);
                    vs.z = 'z';
                    
                    times.z = Date.now();
                    var dt = times.z - times.x;
                    assert.ok(dt >= 50 && dt < 75);
                })
            ;
        })
        .do(function (vars) {
            vars.y = 'y';
            order.push(4);
            
            times.y = Date.now();
        })
        .wait(100)
        .do(function (vars) {
            assert.eql(order, [1,2,3,4]);
            assert.eql(vars, { x : 'x', y : 'y', z : 'z' });
            clearTimeout(to);
            
            times.end = Date.now();
            var dt = times.end - times.y;
            assert.ok(dt >= 100 && dt < 125)
        })
    ;
};

exports.nestNext = function () {
    var ch = (function () {
        var vars = {};
        return Chainsaw(function (saw) {
            this.do = function (cb) {
                saw.nest(false, function () {
                    var args = [].slice.call(arguments);
                    args.push(saw.next);
                    cb.apply(this, args);
                }, vars);
            };
        });
    })();
    
    var order = [];
    var to = setTimeout(function () {
        assert.fail("Didn't get to the end");
    }, 500);
    
    var times = [];
    
    ch
        .do(function (vars, next_) {
            vars.x = 'y';
            order.push(1);
            
            this
                .do(function (vs, next) {
                    order.push(2);
                    vs.x = 'x';
                    setTimeout(next, 30);
                })
                .do(function (vs, next) {
                    order.push(3);
                    vs.z = 'z';
                    setTimeout(next, 10);
                })
                .do(function () {
                    setTimeout(next_, 20);
                })
            ;
        })
        .do(function (vars, next) {
            vars.y = 'y';
            order.push(4);
            setTimeout(next, 5);
        })
        .do(function (vars) {
            assert.eql(order, [1,2,3,4]);
            assert.eql(vars, { x : 'x', y : 'y', z : 'z' });
            
            clearTimeout(to);
        })
    ;
};

exports.builder = function () {
    var cx = Chainsaw(function (saw) {
        this.x = function () {};
    });
    assert.ok(cx.x);
    
    var cy = Chainsaw(function (saw) {
        return { y : function () {} };
    });
    assert.ok(cy.y);
    
    var cz = Chainsaw(function (saw) {
        return { z : function (cb) { saw.nest(cb) } };
    });
    assert.ok(cz.z);
    
    var to = setTimeout(function () {
        assert.fail("Nested z didn't run");
    }, 50);
    
    cz.z(function () {
        clearTimeout(to);
        assert.ok(this.z);
    });
};

this.attr = function () {
    var to = setTimeout(function () {
        assert.fail("attr chain didn't finish");
    }, 50);
    
    var xy = [];
    var ch = Chainsaw(function (saw) {
        this.h = {
            x : function () { 
                xy.push('x');
                saw.next();
            },
            y : function () {
                xy.push('y');
                saw.next();
                assert.eql(xy, ['x','y']);
                clearTimeout(to);
            }
        };
    });
    assert.ok(ch.h);
    assert.ok(ch.h.x);
    assert.ok(ch.h.y);
    
    ch.h.x().h.y();
};

exports.down = function () {
    var error = null;
    var s;
    var ch = Chainsaw(function (saw) {
        s = saw;
        this.raise = function (err) {
            error = err;
            saw.down('catch');
        };
        
        this.do = function (cb) {
            cb.call(this);
        };
        
        this.catch = function (cb) {
            if (error) {
                saw.nest(cb, error);
                error = null;
            }
            else saw.next();
        };
    });
    
    var to = setTimeout(function () {
        assert.fail(".do() after .catch() didn't fire");
    }, 50);
    
    ch
        .do(function () {
            this.raise('pow');
        })
        .do(function () {
            assert.fail("raise didn't skip over this do block");
        })
        .catch(function (err) {
            assert.equal(err, 'pow');
        })
        .do(function () {
            clearTimeout(to);
        })
    ;
};

exports.trap = function () {
    var error = null;
    var ch = Chainsaw(function (saw) {
        var pars = 0;
        var stack = [];
        var i = 0;
        
        this.par = function (cb) {
            pars ++;
            var j = i ++;
            cb.call(function () {
                pars --;
                stack[j] = [].slice.call(arguments);
                saw.down('result');
            });
            saw.next();
        };
        
        this.join = function (cb) {
            saw.trap('result', function () {
                if (pars == 0) {
                    cb.apply(this, stack);
                    saw.next();
                }
            });
        };
        
        this.raise = function (err) {
            error = err;
            saw.down('catch');
        };
        
        this.do = function (cb) {
            cb.call(this);
        };
        
        this.catch = function (cb) {
            if (error) {
                saw.nest(cb, error);
                error = null;
            }
            else saw.next();
        };
    });
    
    var to = setTimeout(function () {
        assert.fail(".do() after .join() didn't fire");
    }, 100);
    var tj = setTimeout(function () {
        assert.fail('.join() never fired');
    }, 100);
    
    var joined = false;
    ch
        .par(function () {
            setTimeout(this.bind(null, 1), 50);
        })
        .par(function () {
            setTimeout(this.bind(null, 2), 25);
        })
        .join(function (x, y) {
            assert.equal(x[0], 1);
            assert.equal(y[0], 2);
            clearTimeout(tj);
            joined = true;
        })
        .do(function () {
            clearTimeout(to);
            assert.ok(joined);
        })
    ;
};

exports.jump = function () {
    var to = setTimeout(function () {
        assert.fail('builder never fired');
    }, 50);
    
    var xs = [ 4, 5, 6, -4, 8, 9, -1, 8 ];
    var xs_ = [];
    
    var ch = Chainsaw(function (saw) {
        this.x = function (i) {
            xs_.push(i);
            saw.next();
        };
        
        this.y = function (step) {
            var x = xs.shift();
            if (x > 0) saw.jump(step);
            else saw.next();
        };
        
        saw.on('end', function () {
            clearTimeout(to);
            assert.eql(xs, [ 8 ]);
            assert.eql(xs_, [ 1, 1, 1, 1, 2, 3, 2, 3, 2, 3 ]);
        });
    });
    
    ch
        .x(1)
        .y(0)
        .x(2)
        .x(3)
        .y(2)
    ;
};
