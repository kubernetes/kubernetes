var feature, supported = [], unsupported = [];

phantom.injectJs('modernizr.js');
console.log('Detected features (using Modernizr ' + Modernizr._version + '):');
for (feature in Modernizr) {
    if (Modernizr.hasOwnProperty(feature)) {
        if (feature[0] !== '_' && typeof Modernizr[feature] !== 'function' &&
            feature !== 'input' && feature !== 'inputtypes') {
            if (Modernizr[feature]) {
                supported.push(feature);
            } else {
                unsupported.push(feature);
            }
        }
    }
}

console.log('');
console.log('Supported:');
supported.forEach(function (e) {
    console.log('  ' + e);
});

console.log('');
console.log('Not supported:');
unsupported.forEach(function (e) {
    console.log('  ' + e);
});
phantom.exit();

