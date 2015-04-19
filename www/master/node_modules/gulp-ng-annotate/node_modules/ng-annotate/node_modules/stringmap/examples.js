var StringMap = require("./stringmap");

var sm1 = new StringMap();
sm1.set("greeting", "yoyoma");
sm1.set("check", true);
sm1.set("__proto__", -1);
console.log(sm1.has("greeting")); // true
console.log(sm1.get("__proto__")); // -1
sm1.remove("greeting");
console.log(sm1.keys()); // [ 'check', '__proto__' ]
console.log(sm1.values()); // [ true, -1 ]
console.log(sm1.items()); // [ [ 'check', true ], [ '__proto__', -1 ] ]
console.log(sm1.toString()); // {"check":true,"__proto__":-1}

var sm2 = new StringMap({
    one: 1,
    two: 2,
});
console.log(sm2.map(function(value, key) {
    return value * value;
})); // [ 1, 4 ]
sm2.forEach(function(value, key) {
    // ...
});
console.log(sm2.isEmpty()); // false
console.log(sm2.size()); // 2

var sm3 = sm1.clone();
sm3.merge(sm2);
sm3.setMany({
    a: {},
    b: [],
});
console.log(sm3.toString()); // {"check":true,"one":1,"two":2,"a":{},"b":[],"__proto__":-1}
