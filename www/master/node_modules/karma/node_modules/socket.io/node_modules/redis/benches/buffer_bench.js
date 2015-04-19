var source = new Buffer(100),
    dest = new Buffer(100), i, j, k, tmp, count = 1000000, bytes = 100;
    
for (i = 99 ; i >= 0 ; i--) {
    source[i] = 120;
}

var str = "This is a nice String.",
    buf = new Buffer("This is a lovely Buffer.");

var start = new Date();
for (i = count * 100; i > 0 ; i--) {
    if (Buffer.isBuffer(str)) {}
}
var end = new Date();
console.log("Buffer.isBuffer(str) " + (end - start) + " ms");

var start = new Date();
for (i = count * 100; i > 0 ; i--) {
    if (Buffer.isBuffer(buf)) {}
}
var end = new Date();
console.log("Buffer.isBuffer(buf) " + (end - start) + " ms");

var start = new Date();
for (i = count * 100; i > 0 ; i--) {
    if (str instanceof Buffer) {}
}
var end = new Date();
console.log("str instanceof Buffer " + (end - start) + " ms");

var start = new Date();
for (i = count * 100; i > 0 ; i--) {
    if (buf instanceof Buffer) {}
}
var end = new Date();
console.log("buf instanceof Buffer " + (end - start) + " ms");

for (i = bytes ; i > 0 ; i --) {
    var start = new Date();
    for (j = count ; j > 0; j--) {
        tmp = source.toString("ascii", 0, bytes);
    }
    var end = new Date();
    console.log("toString() " + i + " bytes " + (end - start) + " ms");
}

for (i = bytes ; i > 0 ; i --) {
    var start = new Date();
    for (j = count ; j > 0; j--) {
        tmp = "";
        for (k = 0; k <= i ; k++) {
            tmp += String.fromCharCode(source[k]);
        }
    }
    var end = new Date();
    console.log("manual string " + i + " bytes " + (end - start) + " ms");
}

for (i = bytes ; i > 0 ; i--) {
    var start = new Date();
    for (j = count ; j > 0 ; j--) {
        for (k = i ; k > 0 ; k--) {
            dest[k] = source[k];
        }
    }
    var end = new Date();
    console.log("Manual copy " + i + " bytes " + (end - start) + " ms");
}

for (i = bytes ; i > 0 ; i--) {
    var start = new Date();
    for (j = count ; j > 0 ; j--) {
        for (k = i ; k > 0 ; k--) {
            dest[k] = 120;
        }
    }
    var end = new Date();
    console.log("Direct assignment " + i + " bytes " + (end - start) + " ms");
}

for (i = bytes ; i > 0 ; i--) {
    var start = new Date();
    for (j = count ; j > 0 ; j--) {
        source.copy(dest, 0, 0, i);
    }
    var end = new Date();
    console.log("Buffer.copy() " + i + " bytes " + (end - start) + " ms");
}
