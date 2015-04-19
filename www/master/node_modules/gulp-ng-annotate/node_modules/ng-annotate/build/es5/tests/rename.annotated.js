"use strict";

// long form
angular.module("MyMod").controller("MyCtrl", ["$aRenamed", "$bRenamed", function($a, $b) {
}]);

// w/ dependencies
angular.module("MyMod", ["OtherMod"]).controller("MyCtrl", ["$cRenamed", "$dRenamed", "$eRenamed", "$fRenamed", "$gRenamed", "$hRenamed", "$iRenamed", function($c, $d, $e, $f, $g, $h, $i) {
}]);

// simple
myMod.service("$aRenamed", ["$bRenamed", function($b) {
}]);
myMod.controller("foo", ["$aRenamed", "$bRenamed", function($a, $b) {
}]);
myMod.service("foo", ["$cRenamed", "$dRenamed", function($c, $d) {
}]);
myMod.factory("foo", ["$eRenamed", "$fRenamed", function($e, $f) {
}]);
myMod.directive("foo", ["$gRenamed", "$hRenamed", function($g, $h) {
}]);
myMod.filter("foo", ["$iRenamed", "$aRenamed", function($i, $a) {
}]);
myMod.animation("foo", ["$bRenamed", "$cRenamed", function($b, $c) {
}]);
myMod.invoke("foo", ["$dRenamed", "$eRenamed", function($d, $e) {
}]);

// implicit config function
angular.module("MyMod", ["OtherMod"], ["$interpolateProvider", function($interpolateProvider) {}]).controller("foo", ["$fRenamed", function($f) {}]);

// object property
var myObj = {};
myObj.myMod = angular.module("MyMod");
myObj.myMod.controller("foo", ["$gRenamed", "$hRenamed", function($g, $h) { a }]);

// run, config don't take names
myMod.run(["$iRenamed", "$aRenamed", function($i, $a) {
}]);
angular.module("MyMod").run(["$bRenamed", function($b) {
}]);
myMod.config(["$cRenamed", "$dRenamed", function($c, $d) {
}]);
angular.module("MyMod").config(function() {
});

// directive return object
myMod.directive("foo", ["$eRenamed", function($e) {
    return {
        controller: ["$fRenamed", "$gRenamed", function($f, $g) {
            bar;
        }]
    }
}]);
myMod.directive("foo", ["$hRenamed", function($h) {
    return {
        controller: function() {
            bar;
        }
    }
}]);

// provider, provider $get
myMod.provider("foo", ["$iRenamed", function($i) {
    this.$get = ["$aRenamed", "$bRenamed", function($a, $b) {
        bar;
    }];
    self.$get = ["$cRenamed", function($c) {}];
    that.$get = ["$dRenamed", function($d) {}];
    ignore.$get = function($e) {};
}]);
myMod.provider("foo", function() {
    this.$get = ["$fRenamed", function($f) {
        bar;
    }];
});
myMod.provider("foo", function() {
    return {
        $get: ["$gRenamed", "$hRenamed", function($g, $h) {
            bar;
        }]};
});
myMod.provider("foo", {
    $get: ["$iRenamed", function($i) {
        bar;
    }]
});
