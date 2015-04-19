module.exports = {
  override: function (source1, source2) {
    var target = {};
    for (var key1 in source1)
      target[key1] = source1[key1];
    for (var key2 in source2)
      target[key2] = source2[key2];

    return target;
  }
};
