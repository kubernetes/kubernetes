module.exports = function LongToShortHex(data) {
  return {
    process: function() {
      return data.replace(/([,: \(])#([0-9a-f]{6})/gi, function(match, prefix, color) {
        if (color[0] == color[1] && color[2] == color[3] && color[4] == color[5])
          return prefix + '#' + color[0] + color[2] + color[4];
        else
          return prefix + '#' + color;
      });
    }
  };
};
