(function (tree) {
//
// RGB Colors - #ff0014, #eee
//
tree.Color = function (rgb, a) {
    //
    // The end goal here, is to parse the arguments
    // into an integer triplet, such as `128, 255, 0`
    //
    // This facilitates operations and conversions.
    //
    if (Array.isArray(rgb)) {
        this.rgb = rgb;
    } else if (rgb.length == 6) {
        this.rgb = rgb.match(/.{2}/g).map(function (c) {
            return parseInt(c, 16);
        });
    } else {
        this.rgb = rgb.split('').map(function (c) {
            return parseInt(c + c, 16);
        });
    }
    this.alpha = typeof(a) === 'number' ? a : 1;
};

var transparentKeyword = "transparent";

tree.Color.prototype = {
    type: "Color",
    eval: function () { return this; },
    luma: function () {
        var r = this.rgb[0] / 255,
            g = this.rgb[1] / 255,
            b = this.rgb[2] / 255;

        r = (r <= 0.03928) ? r / 12.92 : Math.pow(((r + 0.055) / 1.055), 2.4);
        g = (g <= 0.03928) ? g / 12.92 : Math.pow(((g + 0.055) / 1.055), 2.4);
        b = (b <= 0.03928) ? b / 12.92 : Math.pow(((b + 0.055) / 1.055), 2.4);

        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    },

    genCSS: function (env, output) {
        output.add(this.toCSS(env));
    },
    toCSS: function (env, doNotCompress) {
        var compress = env && env.compress && !doNotCompress,
            alpha = tree.fround(env, this.alpha);

        // If we have some transparency, the only way to represent it
        // is via `rgba`. Otherwise, we use the hex representation,
        // which has better compatibility with older browsers.
        // Values are capped between `0` and `255`, rounded and zero-padded.
        if (alpha < 1) {
            if (alpha === 0 && this.isTransparentKeyword) {
                return transparentKeyword;
            }
            return "rgba(" + this.rgb.map(function (c) {
                return clamp(Math.round(c), 255);
            }).concat(clamp(alpha, 1))
                .join(',' + (compress ? '' : ' ')) + ")";
        } else {
            var color = this.toRGB();

            if (compress) {
                var splitcolor = color.split('');

                // Convert color to short format
                if (splitcolor[1] === splitcolor[2] && splitcolor[3] === splitcolor[4] && splitcolor[5] === splitcolor[6]) {
                    color = '#' + splitcolor[1] + splitcolor[3] + splitcolor[5];
                }
            }

            return color;
        }
    },

    //
    // Operations have to be done per-channel, if not,
    // channels will spill onto each other. Once we have
    // our result, in the form of an integer triplet,
    // we create a new Color node to hold the result.
    //
    operate: function (env, op, other) {
        var rgb = [];
        var alpha = this.alpha * (1 - other.alpha) + other.alpha;
        for (var c = 0; c < 3; c++) {
            rgb[c] = tree.operate(env, op, this.rgb[c], other.rgb[c]);
        }
        return new(tree.Color)(rgb, alpha);
    },

    toRGB: function () {
        return toHex(this.rgb);
    },

    toHSL: function () {
        var r = this.rgb[0] / 255,
            g = this.rgb[1] / 255,
            b = this.rgb[2] / 255,
            a = this.alpha;

        var max = Math.max(r, g, b), min = Math.min(r, g, b);
        var h, s, l = (max + min) / 2, d = max - min;

        if (max === min) {
            h = s = 0;
        } else {
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2;               break;
                case b: h = (r - g) / d + 4;               break;
            }
            h /= 6;
        }
        return { h: h * 360, s: s, l: l, a: a };
    },
    //Adapted from http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
    toHSV: function () {
        var r = this.rgb[0] / 255,
            g = this.rgb[1] / 255,
            b = this.rgb[2] / 255,
            a = this.alpha;

        var max = Math.max(r, g, b), min = Math.min(r, g, b);
        var h, s, v = max;

        var d = max - min;
        if (max === 0) {
            s = 0;
        } else {
            s = d / max;
        }

        if (max === min) {
            h = 0;
        } else {
            switch(max){
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return { h: h * 360, s: s, v: v, a: a };
    },
    toARGB: function () {
        return toHex([this.alpha * 255].concat(this.rgb));
    },
    compare: function (x) {
        if (!x.rgb) {
            return -1;
        }
        
        return (x.rgb[0] === this.rgb[0] &&
            x.rgb[1] === this.rgb[1] &&
            x.rgb[2] === this.rgb[2] &&
            x.alpha === this.alpha) ? 0 : -1;
    }
};

tree.Color.fromKeyword = function(keyword) {
    keyword = keyword.toLowerCase();

    if (tree.colors.hasOwnProperty(keyword)) {
        // detect named color
        return new(tree.Color)(tree.colors[keyword].slice(1));
    }
    if (keyword === transparentKeyword) {
        var transparent = new(tree.Color)([0, 0, 0], 0);
        transparent.isTransparentKeyword = true;
        return transparent;
    }
};

function toHex(v) {
    return '#' + v.map(function (c) {
        c = clamp(Math.round(c), 255);
        return (c < 16 ? '0' : '') + c.toString(16);
    }).join('');
}

function clamp(v, max) {
    return Math.min(Math.max(v, 0), max); 
}

})(require('../tree'));
