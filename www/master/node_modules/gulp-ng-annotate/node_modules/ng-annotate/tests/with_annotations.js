"use strict";

// long form
angular.module("MyMod").controller("MyCtrl", ["$scope", "$timeout", function($scope, $timeout) {
}]);

// w/ dependencies
angular.module("MyMod", ["OtherMod"]).controller("MyCtrl", ["$scope", "$timeout", function($scope, $timeout) {
}]);

// simple
myMod.controller("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);
myMod.service("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);
myMod.factory("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);
myMod.directive("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);
myMod.filter("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);
myMod.animation("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);
myMod.invoke("foo", ["$scope", "$timeout", function($scope, $timeout) {
}]);

// implicit config function
angular.module("MyMod", ["$interpolateProvider", function($interpolateProvider) {}]);
angular.module("MyMod", ["OtherMod"], ["$interpolateProvider", function($interpolateProvider) {}]);
angular.module("MyMod", ["OtherMod"], ["$interpolateProvider", function($interpolateProvider) {}]).controller("foo", ["$scope", function($scope) {}]);

// object property
var myObj = {};
myObj.myMod = angular.module("MyMod");
myObj.myMod.controller("foo", ["$scope", "$timeout", function($scope, $timeout) { a }]);

// no dependencies => no need to wrap the function in an array
myMod.controller("foo", function() {
});
myMod.service("foo", function() {
});
myMod.factory("foo", function() {
});
myMod.directive("foo", function() {
});
myMod.filter("foo", function() {
});
myMod.animation("foo", function() {
});
myMod.invoke("foo", function() {
});

// run, config don't take names
myMod.run(["$scope", "$timeout", function($scope, $timeout) {
}]);
angular.module("MyMod").run(["$scope", function($scope) {
}]);
myMod.config(["$scope", "$timeout", function($scope, $timeout) {
}]);
angular.module("MyMod").config(function() {
});

// directive return object
myMod.directive("foo", ["$scope", function($scope) {
    return {
        controller: ["$scope", "$timeout", function($scope, $timeout) {
            bar;
        }]
    }
}]);
myMod.directive("foo", ["$scope", function($scope) {
    return {
        controller: function() {
            bar;
        }
    }
}]);

// provider, provider $get
myMod.provider("foo", ["$scope", function($scope) {
    this.$get = ["$scope", "$timeout", function($scope, $timeout) {
        bar;
    }];
    self.$get = ["$scope", function($scope) {}];
    that.$get = ["$scope", function($scope) {}];
    ignore.$get = function($scope) {};
}]);
myMod.provider("foo", function() {
    this.$get = function() {
        bar;
    };
});
myMod.provider("foo", function() {
    return {
        $get: ["$scope", "$timeout", function($scope, $timeout) {
            bar;
        }]};
});
myMod.provider("foo", function() {
    return {
        $get: function() {
            bar;
        }};
});
myMod.provider("foo", {
    $get: ["$scope", "$timeout", function($scope, $timeout) {
        bar;
    }]
});
myMod.provider("foo", {
    $get: function() {
        bar;
    }
});
myMod.provider("foo", {
    "$get": ["$scope", "$timeout", function($scope, $timeout) {
        bar;
    }]
});
myMod.provider("foo", {
    '$get': ["$scope", "$timeout", function($scope, $timeout) {
        bar;
    }]
});

myMod.provider("foo", ["x", function(x) {
    this.$get = ["a", "b", function(a,b) {}];
}]);

myMod.provider("foo", extprov);
function extprov(x) {
    this.$get = ["a", "b", function(a,b) {}];
    this.$get = fooget;
    this.$get = inner;

    function inner(c, d) {
    }
    inner.$inject = ["c", "d"];
}
extprov.$inject = ["x"];

function fooget(b) {
    this.$get = fooget2;
}
fooget.$inject = ["b"];

function fooget2(c) {
}
fooget2.$inject = ["c"];

// chaining
myMod.directive("foo", ["$a", "$b", function($a, $b) {
    a;
}]).factory("foo", function() {
        b;
    }).config(["$c", function($c) {
        c;
    }]).filter("foo", ["$d", "$e", function($d, $e) {
        d;
    }]).animation("foo", ["$f", "$g", function($f, $g) {
        e;
    }]);

angular.module("MyMod").directive("foo", ["$a", "$b", function($a, $b) {
    a;
}]).provider("foo", function() {
        return {
            $get: ["$scope", "$timeout", function($scope, $timeout) {
                bar;
            }]};
    }).value("foo", "bar")
    .constant("foo", "bar")
    .bootstrap(element, [], {})
    .factory("foo", function() {
        b;
    }).config(["$c", function($c) {
        c;
    }]).filter("foo", ["$d", "$e", function($d, $e) {
        d;
    }]).animation("foo", ["$f", "$g", function($f, $g) {
        e;
    }]).invoke("foo", ["$h", "$i", function($h, $i) {
        f;
    }]);

// $provide
angular.module("myMod").controller("foo", function() {
    $provide.decorator("foo", ["$scope", function($scope) {}]);
    $provide.service("foo", ["$scope", function($scope) {}]);
    $provide.factory("foo", ["$scope", function($scope) {}]);
    //$provide.provider
    $provide.provider("foo", ["$scope", function($scope) {
        this.$get = ["$scope", function($scope) {}];
        return { $get: ["$scope", "$timeout", function($scope, $timeout) {}]};
    }]);
    $provide.provider("foo", {
        $get: ["$scope", "$timeout", function($scope, $timeout) {}]
    });
});
// negative $provide
function notInContext() {
    $provide.decorator("foo", function($scope) {});
    $provide.service("foo", function($scope) {});
    $provide.factory("foo", function($scope) {});
    $provide.provider("foo", function($scope) {
        this.$get = function($scope) {};
        return { $get: function($scope, $timeout) {}};
    });
    $provide.provider("foo", {
        $get: function($scope, $timeout) {}
    });
}


// all the patterns below matches only when we're inside a detected angular module
angular.module("MyMod").directive("pleasematchthis", function() {

    // $injector.invoke
    $injector.invoke(["$compile", function($compile) {
        $compile(myElement)(scope);
    }]);

    // $httpProvider
    $httpProvider.interceptors.push(["$scope", function($scope) { a }]);
    $httpProvider.responseInterceptors.push(["$scope", function($scope) { a }], ["a", "b", function(a, b) { b }], function() { c });

    // $routeProvider
    $routeProvider.when("path", {
        controller: ["$scope", function($scope) { a }]
    }).when("path2", {
            controller: ["$scope", function($scope) { b }],
            resolve: {
                zero: function() { a },
                more: ["$scope", "$timeout", function($scope, $timeout) { b }],
                something: "else",
            },
            dontAlterMe: function(arg) {},
        });

    // ui-router
    $stateProvider.state("myState", {
        resolve: {
            simpleObj: function() { a },
            promiseObj: ["$scope", "$timeout", function($scope, $timeout) { b }],
            translations: "translations",
        },
        views: {
            viewa: {
                controller: ["$scope", "myParam", function($scope, myParam) {}],
                controllerProvider: ["$stateParams", function($stateParams) {}],
                templateProvider: ["$scope", function($scope) {}],
                dontAlterMe: function(arg) {},
                resolve: {
                    myParam: ["$stateParams", function($stateParams) {
                        return $stateParams.paramFromDI;
                    }]
                },
            },
            viewb: {
                dontAlterMe: function(arg) {},
                templateProvider: ["$scope", function($scope) {}],
                controller: ["$scope", function($scope) {}],
            },
            dontAlterMe: null,
        },
        controller: ["$scope", "simpleObj", "promiseObj", "translations", function($scope, simpleObj, promiseObj, translations) { c }],
        controllerProvider: ["$scope", function($scope) { g }],
        templateProvider: ["$scope", function($scope) { h }],
        onEnter: ["$scope", function($scope) { d }],
        onExit: ["$scope", function($scope) { e }],
        dontAlterMe: function(arg) { f },
    }).state("myState2", {
            controller: ["$scope", function($scope) {}],
        }).state({
            name: "myState3",
            controller: ["$scope", "simpleObj", "promiseObj", "translations", function($scope, simpleObj, promiseObj, translations) { c }],
        });
    $urlRouterProvider.when("/", ["$match", function($match) { a; }]);
    $urlRouterProvider.otherwise("", function(a) { a; });
    $urlRouterProvider.rule(function(a) { a; }).anything().when("/", ["$location", function($location) { a; }]);

    stateHelperProvider.setNestedState({
        controller: ["$scope", "simpleObj", "promiseObj", "translations", function($scope, simpleObj, promiseObj, translations) { c }],

        children: [
            {
                name: "a",
                controller: ["a", function(a) {}],
                resolve: {
                    f: ["$a", function($a) {}],
                },
                children: [
                    {
                        name: "ab",
                        controller: ["ab", function(ab) {}],
                        resolve: {
                            f: ["$ab", function($ab) {}],
                        },
                        children: [
                            {
                                name: "abc",
                                controller: ["abc", function(abc) {}],
                                resolve: {
                                    f: ["$abc", function($abc) {}],
                                },
                            },
                        ],
                    },
                ],
            },
            {
                name: "b",
                controller: ["b", function(b) {}],
                views: {
                    viewa: {
                        controller: ["$scope", "myParam", function($scope, myParam) {}],
                        controllerProvider: ["$stateParams", function($stateParams) {}],
                        templateProvider: ["$scope", function($scope) {}],
                        dontAlterMe: function(arg) {},
                        resolve: {
                            myParam: ["$stateParams", function($stateParams) {
                                return $stateParams.paramFromDI;
                            }]
                        },
                    },
                    viewb: {
                        dontAlterMe: function(arg) {},
                        templateProvider: ["$scope", function($scope) {}],
                        controller: ["$scope", function($scope) {}],
                    },
                    dontAlterMe: null,
                },
            },
        ],
    });
    stateHelperProvider.setNestedState({
        controller: ["$scope", "simpleObj", "promiseObj", "translations", function($scope, simpleObj, promiseObj, translations) { c }],
    }, true);

    // angular ui / ui-bootstrap $modal
    $modal.open({
        templateUrl: "str",
        controller: ["$scope", function($scope) {}],
        resolve: {
            items: ["MyService", function(MyService) {}],
            data: ["a", "b", function(a, b) {}],
            its: 42,
        },
        donttouch: function(me) {},
    });

    // angular material design $mdBottomSheet, $mdDialog, $mdToast
    $mdDialog.show({
        templateUrl: "str",
        controller: ["$scope", function($scope) {}],
        resolve: {
            items: ["MyService", function(MyService) {}],
            data: ["a", "b", function(a, b) {}],
            its: 42,
        },
        donttouch: function(me) {},
    });
    $mdBottomSheet.show({
        templateUrl: "str",
        controller: ["$scope", function($scope) {}],
        resolve: {
            items: ["MyService", function(MyService) {}],
            data: ["a", "b", function(a, b) {}],
            its: 42,
        },
        donttouch: function(me) {},
    });
    $mdToast.show({
        templateUrl: "str",
        controller: ["$scope", function($scope) {}],
        resolve: {
            items: ["MyService", function(MyService) {}],
            data: ["a", "b", function(a, b) {}],
            its: 42,
        },
        donttouch: function(me) {},
    });
});

// none of the patterns below matches because they are not in an angular module context
// this should be a straight copy of the code above, with identical copies in
// with_annotations(_single).js
foobar.irrespective("dontmatchthis", function() {

    // $injector.invoke
    $injector.invoke(function($compile) {
        $compile(myElement)(scope);
    });

    // $httpProvider
    $httpProvider.interceptors.push(function($scope) { a });
    $httpProvider.responseInterceptors.push(function($scope) { a }, function(a, b) { b }, function() { c });

    // $routeProvider
    $routeProvider.when("path", {
        controller: function($scope) { a }
    }).when("path2", {
        controller: function($scope) { b },
        resolve: {
            zero: function() { a },
            more: function($scope, $timeout) { b },
            something: "else",
        },
        dontAlterMe: function(arg) {},
    });

    // ui-router
    $stateProvider.state("myState", {
        resolve: {
            simpleObj: function() { a },
            promiseObj: function($scope, $timeout) { b },
            translations: "translations",
        },
        views: {
            viewa: {
                controller: function($scope, myParam) {},
                controllerProvider: function($stateParams) {},
                templateProvider: function($scope) {},
                dontAlterMe: function(arg) {},
                resolve: {
                    myParam: function($stateParams) {
                        return $stateParams.paramFromDI;
                    }
                },
            },
            viewb: {
                dontAlterMe: function(arg) {},
                templateProvider: function($scope) {},
                controller: function($scope) {},
            },
            dontAlterMe: null,
        },
        controller: function($scope, simpleObj, promiseObj, translations) { c },
        controllerProvider: function($scope) { g },
        templateProvider: function($scope) { h },
        onEnter: function($scope) { d },
        onExit: function($scope) { e },
        dontAlterMe: function(arg) { f },
    }).state("myState2", {
        controller: function($scope) {},
    }).state({
        name: "myState3",
        controller: function($scope, simpleObj, promiseObj, translations) { c },
    });
    $urlRouterProvider.when("/", function($match) { a; });
    $urlRouterProvider.otherwise("", function(a) { a; });
    $urlRouterProvider.rule(function(a) { a; }).anything().when("/", function($location) { a; });

    stateHelperProvider.setNestedState({
        controller: function($scope, simpleObj, promiseObj, translations) { c },

        children: [
            {
                name: "a",
                controller: function(a) {},
                resolve: {
                    f: function($a) {},
                },
                children: [
                    {
                        name: "ab",
                        controller: function(ab) {},
                        resolve: {
                            f: function($ab) {},
                        },
                        children: [
                            {
                                name: "abc",
                                controller: function(abc) {},
                                resolve: {
                                    f: function($abc) {},
                                },
                            },
                        ],
                    },
                ],
            },
            {
                name: "b",
                controller: function(b) {},
                views: {
                    viewa: {
                        controller: function($scope, myParam) {},
                        controllerProvider: function($stateParams) {},
                        templateProvider: function($scope) {},
                        dontAlterMe: function(arg) {},
                        resolve: {
                            myParam: function($stateParams) {
                                return $stateParams.paramFromDI;
                            }
                        },
                    },
                    viewb: {
                        dontAlterMe: function(arg) {},
                        templateProvider: function($scope) {},
                        controller: function($scope) {},
                    },
                    dontAlterMe: null,
                },
            },
        ],
    });
    stateHelperProvider.setNestedState({
        controller: function($scope, simpleObj, promiseObj, translations) { c },
    }, true);

    // angular ui / ui-bootstrap $modal
    $modal.open({
        templateUrl: "str",
        controller: function($scope) {},
        resolve: {
            items: function(MyService) {},
            data: function(a, b) {},
            its: 42,
        },
        donttouch: function(me) {},
    });

    // angular material design $mdBottomSheet, $mdDialog, $mdToast
    $mdDialog.show({
        templateUrl: "str",
        controller: function($scope) {},
        resolve: {
            items: function(MyService) {},
            data: function(a, b) {},
            its: 42,
        },
        donttouch: function(me) {},
    });
    $mdBottomSheet.show({
        templateUrl: "str",
        controller: function($scope) {},
        resolve: {
            items: function(MyService) {},
            data: function(a, b) {},
            its: 42,
        },
        donttouch: function(me) {},
    });
    $mdToast.show({
        templateUrl: "str",
        controller: function($scope) {},
        resolve: {
            items: function(MyService) {},
            data: function(a, b) {},
            its: 42,
        },
        donttouch: function(me) {},
    });
});

// explicit annotations
var x = /* @ngInject */ ["$scope", function($scope) {
}];

var obj = {};
obj.bar = /*@ngInject*/ ["$scope", function($scope) {}];

obj = {
    controller: /*@ngInject*/ ["$scope", function($scope) {}],
};

obj = /*@ngInject*/ {
    foo: ["a", function(a) {}],
    bar: ["b", "c", function(b, c) {}],
    val: 42,
    inner: {
        circle: ["d", function(d) {}],
        alalalala: "long",
    },
    nest: { many: {levels: ["x", function(x) {}]}},
    but: { onlythrough: ["object literals", {donttouch: function(me) {}}]},
};

obj = {
    /*@ngInject*/
    foo: ["a", function(a) {}],
    bar: function(b, c) {},
};

/*@ngInject*/
obj = {
    foo: ["a", function(a) {}],
    bar: ["b", "c", function(b, c) {}],
    val: 42,
    inner: {
        circle: ["d", function(d) {}],
        alalalala: "long",
    },
    nest: { many: {levels: ["x", function(x) {}]}},
    but: { onlythrough: ["object literals", {donttouch: function(me) {}}]},
};

/*@ngInject*/
var obj = {
    foo: ["a", function(a) {}],
    bar: ["b", "c", function(b, c) {}],
    val: 42,
    inner: {
        circle: ["d", function(d) {}],
        alalalala: "long",
    },
    nest: { many: {levels: ["x", function(x) {}]}},
    but: { onlythrough: ["object literals", {donttouch: function(me) {}}]},
};

// @ngInject
function foo($scope) {
}
foo.$inject = ["$scope"];

// @ngInject
// otherstuff
function Foo($scope) {
}
Foo.$inject = ["$scope"];

// @ngInject
// has trailing semicolon
var foo1 = function($scope) {
};
foo1.$inject = ["$scope"];

// @ngInject
// lacks trailing semicolon
var foo2 = function($scope) {
}
foo2.$inject = ["$scope"];

// @ngInject
// has trailing semicolon
bar.foo1 = function($scope) {
};
bar.foo1.$inject = ["$scope"];

// @ngInject
// lacks trailing semicolon
bar.foo2 = function($scope) {
}
bar.foo2.$inject = ["$scope"];

// let's zip-zag indentation to make sure that the $inject array lines up properly
    // @ngInject
    function foo3($scope) {}
    foo3.$inject = ["$scope"];
        // @ngInject
        function foo4($scope) {
        }
        foo4.$inject = ["$scope"];
/* @ngInject */ function foo5($scope) {}
foo5.$inject = ["$scope"];
            /* @ngInject */ function foo6($scope) {
            }
            foo6.$inject = ["$scope"];

    // @ngInject
    var foo7 = function($scope) {
    };
    foo7.$inject = ["$scope"];
        // @ngInject
        var foo8 = function($scope) {};
        foo8.$inject = ["$scope"];
// @ngInject
var foo9 = function($scope) {
}
foo9.$inject = ["$scope"];
            // @ngInject
            var foo10 = function($scope) {}
            foo10.$inject = ["$scope"];

    /* @ngInject */ var foo11 = function($scope) {
    };
    foo11.$inject = ["$scope"];
        /* @ngInject */var foo12 = function($scope) {};
        foo12.$inject = ["$scope"];
/* @ngInject */var foo13 = function($scope) {
}
foo13.$inject = ["$scope"];
            /* @ngInject */var foo14 = function($scope) {}
            foo14.$inject = ["$scope"];


// adding an explicit annotation where it isn't needed should work fine
myMod.controller("foo", /*@ngInject*/ ["$scope", "$timeout", function($scope, $timeout) {
}]);


// troublesome return forces different placement of $inject array
function outer() {
    foo;
    MyCtrl.$inject = ["a"];
    return {
        controller: MyCtrl,
    };

    // @ngInject
    function MyCtrl(a) {
    }
}


// explicit annotations using ngInject() instead of /*@ngInject*/
var x = ngInject(["$scope", function($scope) {}]);

obj = ngInject({
    foo: ["a", function(a) {}],
    bar: ["b", "c", function(b, c) {}],
    val: 42,
    inner: {
        circle: ["d", function(d) {}],
        alalalala: "long",
    },
    nest: { many: {levels: ["x", function(x) {}]}},
    but: { onlythrough: ["object literals", {donttouch: function(me) {}}]},
});


// explicit annotations using "ngInject" Directive Prologue
function Foo2($scope) {
    "ngInject";
}
Foo2.$inject = ["$scope"];

var foos3 = function($scope) {
    // comments are ok before the Directive Prologues
    // and there may be multiple Prologues
    "use strict"; "ngInject";
};
foos3.$inject = ["$scope"];

var dual1 = function(a) { "ngInject" }, dual2 = function(b) { "ngInject" };
dual1.$inject = ["a"];
dual2.$inject = ["b"];

g(["c", function(c) {
    "ngInject"
}]);

// Traceur class output example
// class C {
//     constructor($scope) {
//         "ngInject"
//     }
// }
$traceurRuntime.ModuleStore.getAnonymousModule(function() {
    "use strict";
    var C = function C($scope) {
        "ngInject";
    };
    C.$inject = ["$scope"];
    ($traceurRuntime.createClass)(C, {}, {});
    return {};
});


// suppress false positives with /*@ngNoInject*/, ngNoInject() and "ngNoInject"
myMod.controller("suppressed", /*@ngNoInject*/function($scope) {
});
myMod.controller("suppressed", ngNoInject(function($scope) {
}));
myMod.controller("suppressed", function($scope) {
    "ngNoInject";
});

// works the same as ngInject i.e. reference-following, IIFE-jumping and so on
/*@ngNoInject*/
myMod.controller("suppressed", SupFoo1);
myMod.controller("suppressed", SupFoo2);
myMod.controller("suppressed", SupFoo3);
function SupFoo1($scope) {
    "ngNoInject";
}
/*@ngNoInject*/
function SupFoo2($scope) {
}
var SupFoo3 = ngNoInject(function($scope) {
    "ngNoInject";
});


// snippets that shouldn't fool ng-annotate into generating false positives,
//   whether we're inside an angular module or not
myMod.controller("donttouchme", function() {
    // lo-dash regression that happened in the brief time frame when
    // notes (instad of "notes") would match. see issue #22
    var notesForCurrentPage = _.filter(notes, function (note) {
        return note.page.uid === page.uid;
    });
});

// $get is only valid inside provider
myMod.service("donttouch", function() {
    this.$get = function(me) {
    };
});
myMod.service("donttouch", mefn);
function mefn() {
    this.$get = function(me) {
    };
}

// directive return object is only valid inside directive
myMod.service("donttouch", function() {
    return {
        controller: function($scope, $timeout) {
            bar;
        }
    }
});

myMod.directive("donttouch", function() {
    foo.decorator("me", function($scope) {
    });
});

// IIFE-jumping (primarily for compile-to-JS langs)
angular.module("MyMod").directive("foo", ["$a", "$b", function($a, $b) {
    $modal.open({
        resolve: {
            collection: (function(_this) {
                return ["$c", function($c) {
                }];
            })(this),
        },
    });
}]);

var x = /*@ngInject*/ (function() {
    return ["$a", function($a) {
    }];
})();


// IIFE-jumping with reference support
var myCtrl = (function () {
    return function($scope) {
    };
})();
myCtrl.$inject = ["$scope"];
angular.module("MyMod").controller("MyCtrl", myCtrl);


// advanced IIFE-jumping (with reference support)
var myCtrl10 = (function() {
    "use strict";
    // the return statement can appear anywhere on the functions topmost level,
    // including before the myCtrl function definition
    myCtrl.$inject = ["$scope"];
    return myCtrl;
    function myCtrl($scope) {
        foo;
    }
    post;
})();
angular.module("MyMod").controller("MyCtrl", myCtrl10);

var myCtrl11 = (function() {
    pre;
    var myCtrl = function($scope) {
        foo;
    };
    myCtrl.$inject = ["$scope"];
    mid;
    // the return statement can appear anywhere on the functions topmost level,
    // including before the myCtrl function definition
    return myCtrl;
    post;
})();
angular.module("MyMod").controller("MyCtrl", myCtrl11);


// reference support
function MyCtrl1(a, b) {
}
MyCtrl1.$inject = ["a", "b"];
if (true) {
    // proper scope analysis including shadowing
    let MyCtrl1 = function(c) {
    };
    MyCtrl1.$inject = ["c"];
    angular.module("MyMod").directive("foo", MyCtrl1);
}
angular.module("MyMod").controller("bar", MyCtrl1);
function MyCtrl2(z) {
}
MyCtrl2.$inject = ["z"];
funcall(/*@ngInject*/ MyCtrl2); // explicit annotation on reference flows back to definition

angular.module("MyMod").directive("foo", MyDirective);

function MyDirective($stateProvider) {
    $stateProvider.state('astate', {
        resolve: {
            yoyo: ["ma", function(ma) {
            }],
        }
    });
}
MyDirective.$inject = ["$stateProvider"];

/* @ngInject */
function MyDirective2($stateProvider) {
    $stateProvider.state('astate', {
        resolve: {
            yoyo: ["ma", function(ma) {
            }],
        }
    });
}
MyDirective2.$inject = ["$stateProvider"];

// issue 84
(function() {
    var MyCtrl = function($someDependency) {};
    MyCtrl.$inject = ["$someDependency"];
    angular.module('myApp').controller("MyCtrl", MyCtrl);
    MyCtrl.prototype.someFunction = function() {};
})();

// empty var declarator
var MyCtrl12;
angular.module("MyMod").controller('MyCtrl', MyCtrl12);

// issue 115
module.exports = function() {
    "use strict";
    return {
        restrict: 'E',
        replace: true,
        scope: { },
        controller: /*@ngInject*/["$scope", "myService", function($scope, myService) {
        }],
        templateUrl: "mytemplate"
    };
};
