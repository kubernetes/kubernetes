  function firstOnly(x) {
    if (x.length === 0) { throw new EmptyError(); }
    return x[0];
  }
