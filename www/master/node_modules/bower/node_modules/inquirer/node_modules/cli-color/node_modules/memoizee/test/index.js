'use strict';

var aFrom    = require('es5-ext/array/from')
  , nextTick = require('next-tick')

  , join = Array.prototype.join;

module.exports = function (t, a) {
	return {
		"0": function () {
			var i = 0, fn = function () { ++i; return 3; };

			fn = t(fn);
			a(fn(), 3, "First");
			a(fn(1), 3, "Second");
			a(fn(5), 3, "Third");
			a(i, 1, "Called once");
		},
		"1": function () {
			var i = 0, fn = function (x) { ++i; return x; };

			fn = t(fn);
			return {
				"No arg": function () {
					i = 0;
					a(fn(), undefined, "First");
					a(fn(), undefined, "Second");
					a(fn(), undefined, "Third");
					a(i, 1, "Called once");
				},
				Arg: function () {
					var x = {};
					i = 0;
					a(fn(x, 8), x, "First");
					a(fn(x, 4), x, "Second");
					a(fn(x, 2), x, "Third");
					a(i, 1, "Called once");
				},
				"Other Arg": function () {
					var x = {};
					i = 0;
					a(fn(x, 2), x, "First");
					a(fn(x, 9), x, "Second");
					a(fn(x, 3), x, "Third");
					a(i, 1, "Called once");
				}
			};
		},
		"3": function () {
			var i = 0, fn = function (x, y, z) { ++i; return [x, y, z]; }, r;

			fn = t(fn);
			return {
				"No args": function () {
					i = 0;
					a.deep(r = fn(), [undefined, undefined, undefined], "First");
					a(fn(), r, "Second");
					a(fn(), r, "Third");
					a(i, 1, "Called once");
				},
				"Some Args": function () {
					var x = {};
					i = 0;
					a.deep(r = fn(x, 8), [x, 8, undefined], "First");
					a(fn(x, 8), r, "Second");
					a(fn(x, 8), r, "Third");
					a(i, 1, "Called once");
					return {
						Other: function () {
							a.deep(r = fn(x, 5), [x, 5, undefined], "Second");
							a(fn(x, 5), r, "Third");
							a(i, 2, "Called once");
						}
					};
				},
				"Full stuff": function () {
					var x = {};
					i = 0;
					a.deep(r = fn(x, 8, 23, 98), [x, 8, 23], "First");
					a(fn(x, 8, 23, 43), r, "Second");
					a(fn(x, 8, 23, 9), r, "Third");
					a(i, 1, "Called once");
					return {
						Other: function () {
							a.deep(r = fn(x, 23, 8, 13), [x, 23, 8], "Second");
							a(fn(x, 23, 8, 22), r, "Third");
							a(i, 2, "Called once");
						}
					};
				}
			};
		},
		"Normalizer function": function () {
			var i = 0, fn = function () { ++i; return join.call(arguments, '|'); }, mfn;
			mfn = t(fn, { normalizer: function (args) { return Boolean(args[0]); } });
			a(mfn(false, 'raz'), 'false|raz', "#1");
			a(mfn(0, 'dwa'), 'false|raz', "#2");
			a(i, 1, "Called once");
			a(mfn(34, 'bar'), '34|bar', "#3");
			a(i, 2, "Called twice");
			a(mfn(true, 'ola'), '34|bar', "#4");
			a(i, 2, "Called twice #2");
		},
		Dynamic: function () {
			var i = 0, fn = function () { ++i; return arguments; }, r;

			fn = t(fn, { length: false });
			return {
				"No args": function () {
					i = 0;
					a.deep(aFrom(r = fn()), [], "First");
					a(fn(), r, "Second");
					a(fn(), r, "Third");
					a(i, 1, "Called once");
				},
				"Some Args": function () {
					var x = {};
					i = 0;
					a.deep(aFrom(r = fn(x, 8)), [x, 8], "First");
					a(fn(x, 8), r, "Second");
					a(fn(x, 8), r, "Third");
					a(i, 1, "Called once");
				},
				"Many args": function () {
					var x = {};
					i = 0;
					a.deep(aFrom(r = fn(x, 8, 23, 98)), [x, 8, 23, 98], "First");
					a(fn(x, 8, 23, 98), r, "Second");
					a(fn(x, 8, 23, 98), r, "Third");
					a(i, 1, "Called once");
				}
			};
		},
		Resolvers: function () {
			var i = 0, fn, r;
			fn = t(function () { ++i; return arguments; },
				{ length: 3, resolvers: [Boolean, String] });
			return {
				"No args": function () {
					i = 0;
					a.deep(aFrom(r = fn()), [false, 'undefined'], "First");
					a(fn(), r, "Second");
					a(fn(), r, "Third");
					a(i, 1, "Called once");
				},
				"Some Args": function () {
					var x = {};
					i = 0;
					a.deep(aFrom(r = fn(0, 34, x, 45)), [false, '34', x, 45],
						"First");
					a(fn(0, 34, x, 22), r, "Second");
					a(fn(0, 34, x, false), r, "Third");
					a(i, 1, "Called once");
					return {
						Other: function () {
							a.deep(aFrom(r = fn(1, 34, x, 34)),
								[true, '34', x, 34], "Second");
							a(fn(1, 34, x, 89), r, "Third");
							a(i, 2, "Called once");
						}
					};
				}
			};
		},
		"Clear Cache": {
			Specific: function () {
				var i = 0, fn, mfn, x = {};

				fn = function (a, b, c) {
					if (c === 3) {
						++i;
					}
					return arguments;
				};

				mfn = t(fn);
				mfn(1, x, 3);
				mfn(1, x, 4);
				mfn.delete(1, x, 4);
				mfn(1, x, 3);
				mfn(1, x, 3);
				a(i, 1, "Pre clear");
				mfn.delete(1, x, 3);
				mfn(1, x, 3);
				a(i, 2, "After clear");

				i = 0;
				mfn = t(fn, { length: false });
				mfn(1, x, 3);
				mfn(1, x, 3);
				mfn();
				mfn();
				mfn.delete();
				mfn(1, x, 3);
				a(i, 1, "Proper no arguments clear");
			},
			All: function () {
				var i = 0, fn, x = {};

				fn = function () {
					++i;
					return arguments;
				};

				fn = t(fn, { length: 3 });
				fn(1, x, 3);
				fn(1, x, 4);
				fn(1, x, 3);
				fn(1, x, 4);
				a(i, 2, "Pre clear");
				fn.clear();
				fn(1, x, 3);
				fn(1, x, 4);
				fn(1, x, 3);
				fn(1, x, 4);
				a(i, 4, "After clear");
			}
		},
		Primitive: {
			"No args": function (a) {
				var i = 0, fn = function () { ++i; return arguments[0]; }, mfn;
				mfn = t(fn, { primitive: true });
				a(mfn('ble'), 'ble', "#1");
				a(mfn({}), 'ble', "#2");
				a(i, 1, "Called once");
			},
			"One arg": function (a) {
				var i = 0, fn = function (x) { ++i; return x; }, mfn
				  , y = { toString: function () { return 'foo'; } };
				mfn = t(fn, { primitive: true });
				a(mfn(y), y, "#1");
				a(mfn('foo'), y, "#2");
				a(i, 1, "Called once");
			},
			"Many args": function (a) {
				var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn
				  , y = { toString: function () { return 'foo'; } };
				mfn = t(fn, { primitive: true });
				a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#1");
				a(mfn('foo', 'bar', 'zeta'), 'foobarzeta', "#2");
				a(i, 1, "Called once");
			},
			"Clear cache": function (a) {
				var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn
				  , y = { toString: function () { return 'foo'; } };
				mfn = t(fn, { primitive: true });
				a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#1");
				a(mfn('foo', 'bar', 'zeta'), 'foobarzeta', "#2");
				a(i, 1, "Called once");
				mfn.delete('foo', { toString: function () { return 'bar'; } },
					'zeta');
				a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#3");
				a(i, 2, "Called twice");
			}
		},
		"Reference counter": {
			Regular: function (a) {
				var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn;
				mfn = t(fn, { refCounter: true });
				a(mfn.deleteRef(3, 5, 7), null, "Clear before");
				a(mfn(3, 5, 7), 15, "Initial");
				a(mfn(3, 5, 7), 15, "Cache");
				a(mfn.deleteRef(3, 5, 7), false, "Clear #1");
				mfn(3, 5, 7);
				a(mfn.deleteRef(3, 5, 7), false, "Clear #2");
				mfn(3, 5, 7);
				a(mfn.deleteRef(3, 5, 7), false, "Clear #3");
				mfn(3, 5, 7);
				a(i, 1, "Not cleared");
				a(mfn.deleteRef(3, 5, 7), false, "Clear #4");
				a(mfn.deleteRef(3, 5, 7), true, "Clear final");
				mfn(3, 5, 7);
				a(i, 2, "Restarted");
				mfn(3, 5, 7);
				a(i, 2, "Cached again");
			},
			Primitive: function (a) {
				var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn;
				mfn = t(fn, { primitive: true, refCounter: true });
				a(mfn.deleteRef(3, 5, 7), null, "Clear before");
				a(mfn(3, 5, 7), 15, "Initial");
				a(mfn(3, 5, 7), 15, "Cache");
				a(mfn.deleteRef(3, 5, 7), false, "Clear #1");
				mfn(3, 5, 7);
				a(mfn.deleteRef(3, 5, 7), false, "Clear #2");
				mfn(3, 5, 7);
				a(mfn.deleteRef(3, 5, 7), false, "Clear #3");
				mfn(3, 5, 7);
				a(i, 1, "Not cleared");
				a(mfn.deleteRef(3, 5, 7), false, "Clear #4");
				a(mfn.deleteRef(3, 5, 7), true, "Clear final");
				mfn(3, 5, 7);
				a(i, 2, "Restarted");
				mfn(3, 5, 7);
				a(i, 2, "Cached again");
			}
		},
		Async: {
			Regular: {
				Success: function (a, d) {
					var mfn, fn, u = {}, i = 0, invoked = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true });

					a(mfn(3, 7, function (err, res) {
						++invoked;
						a.deep([err, res], [null, 10], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						++invoked;
						a.deep([err, res], [null, 10], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						++invoked;
						a.deep([err, res], [null, 13], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						++invoked;
						a.deep([err, res], [null, 10], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						++invoked;
						a.deep([err, res], [null, 13], "Result B #2");
					}), u, "Initial #3");

					nextTick(function () {
						a(i, 2, "Init Called");
						a(invoked, 5, "Cb Called");

						a(mfn(3, 7, function (err, res) {
							++invoked;
							a.deep([err, res], [null, 10], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							++invoked;
							a.deep([err, res], [null, 13], "Again B: Result");
						}), u, "Again B: Initial");

						nextTick(function () {
							a(i, 2, "Init Called #2");
							a(invoked, 7, "Cb Called #2");

							mfn.delete(3, 7);

							a(mfn(3, 7, function (err, res) {
								++invoked;
								a.deep([err, res], [null, 10], "Again: Result");
							}), u, "Again: Initial");
							a(mfn(5, 8, function (err, res) {
								++invoked;
								a.deep([err, res], [null, 13], "Again B: Result");
							}), u, "Again B: Initial");

							nextTick(function () {
								a(i, 3, "Init  After clear");
								a(invoked, 9, "Cb After clear");
								d();
							});
						});
					});
				},
				"Reference counter": function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, refCounter: true });

					a(mfn.deleteRef(3, 7), null, "Clear ref before");

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #2");
					}), u, "Initial #3");

					nextTick(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Again B: Result");
						}), u, "Again B: Initial");

						nextTick(function () {
							a(i, 2, "Again Called #2");

							a(mfn.deleteRef(3, 7), false, "Clear ref #1");
							a(mfn.deleteRef(3, 7), false, "Clear ref #2");
							a(mfn.deleteRef(3, 7), false, "Clear ref #3");
							a(mfn.deleteRef(3, 7), true, "Clear ref Final");

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Again: Result");
							}), u, "Again: Initial");
							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Again B: Result");
							}), u, "Again B: Initial");

							nextTick(function () {
								a(i, 3, "Call After clear");
								d();
							});
						});
					});
				},
				Error: function (a, d) {
					var mfn, fn, u = {}, i = 0, e = new Error("Test");
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(e);
						});
						return u;
					};

					mfn = t(fn, { async: true });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [e, undefined], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [e, undefined], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [e, undefined], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [e, undefined], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [e, undefined], "Result B #2");
					}), u, "Initial #3");

					nextTick(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [e, undefined], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [e, undefined], "Again B: Result");
						}), u, "Again B: Initial");

						nextTick(function () {
							a(i, 4, "Again Called #2");
							d();
						});
					});
				}
			},
			Primitive: {
				Success: function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, primitive: true });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #2");
					}), u, "Initial #3");

					nextTick(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Again B: Result");
						}), u, "Again B: Initial");

						nextTick(function () {
							a(i, 2, "Again Called #2");

							mfn.delete(3, 7);

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Again: Result");
							}), u, "Again: Initial");
							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Again B: Result");
							}), u, "Again B: Initial");

							nextTick(function () {
								a(i, 3, "Call After clear");
								d();
							});
						});
					});
				},
				"Reference counter": function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, primitive: true, refCounter: true });

					a(mfn.deleteRef(3, 7), null, "Clear ref before");

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #2");
					}), u, "Initial #3");

					nextTick(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Again B: Result");
						}), u, "Again B: Initial");

						nextTick(function () {
							a(i, 2, "Again Called #2");

							a(mfn.deleteRef(3, 7), false, "Clear ref #1");
							a(mfn.deleteRef(3, 7), false, "Clear ref #2");
							a(mfn.deleteRef(3, 7), false, "Clear ref #3");
							a(mfn.deleteRef(3, 7), true, "Clear ref Final");

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Again: Result");
							}), u, "Again: Initial");
							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Again B: Result");
							}), u, "Again B: Initial");

							nextTick(function () {
								a(i, 3, "Call After clear");
								d();
							});
						});
					});
				},
				Error: function (a, d) {
					var mfn, fn, u = {}, i = 0, e = new Error("Test");
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(e);
						});
						return u;
					};

					mfn = t(fn, { async: true, primitive: true });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [e, undefined], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [e, undefined], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [e, undefined], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [e, undefined], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [e, undefined], "Result B #2");
					}), u, "Initial #3");

					nextTick(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [e, undefined], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [e, undefined], "Again B: Result");
						}), u, "Again B: Initial");

						nextTick(function () {
							a(i, 4, "Again Called #2");
							d();
						});
					});
				}
			}
		},
		MaxAge: {
			Regular: {
				Sync: function (a, d) {
					var mfn, fn, i = 0;
					fn = function (x, y) {
						++i;
						return x + y;
					};
					mfn = t(fn, { maxAge: 100 });

					a(mfn(3, 7), 10, "Result #1");
					a(i, 1, "Called #1");
					a(mfn(3, 7), 10, "Result #2");
					a(i, 1, "Called #2");
					a(mfn(5, 8), 13, "Result B #1");
					a(i, 2, "Called B #1");
					a(mfn(3, 7), 10, "Result #3");
					a(i, 2, "Called #3");
					a(mfn(5, 8), 13, "Result B #2");
					a(i, 2, "Called B #2");

					setTimeout(function () {
						a(mfn(3, 7), 10, "Result: Wait");
						a(i, 2, "Called: Wait");
						a(mfn(5, 8), 13, "Result: Wait B");
						a(i, 2, "Called: Wait B");

						setTimeout(function () {
							a(mfn(3, 7), 10, "Result: Wait After");
							a(i, 3, "Called: Wait After");
							a(mfn(5, 8), 13, "Result: Wait After B");
							a(i, 4, "Called: Wait After B");

							a(mfn(3, 7), 10, "Result: Wait After #2");
							a(i, 4, "Called: Wait After #2");
							a(mfn(5, 8), 13, "Result: Wait After B #2");
							a(i, 4, "Called: Wait After B #2");
							d();
						}, 100);
					}, 20);
				},
				Async: function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, maxAge: 100 });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #2");
					}), u, "Initial #3");

					setTimeout(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Again B: Result");
						}), u, "Again B: Initial");

						setTimeout(function () {
							a(i, 2, "Again Called #2");

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Again: Result");
							}), u, "Again: Initial");
							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Again B: Result");
							}), u, "Again B: Initial");

							nextTick(function () {
								a(i, 4, "Call After clear");
								d();
							});
						}, 100);
					}, 20);
				}
			},
			Primitive: {
				Sync: function (a, d) {
					var mfn, fn, i = 0;
					fn = function (x, y) {
						++i;
						return x + y;
					};
					mfn = t(fn, { primitive: true, maxAge: 100 });

					a(mfn(3, 7), 10, "Result #1");
					a(i, 1, "Called #1");
					a(mfn(3, 7), 10, "Result #2");
					a(i, 1, "Called #2");
					a(mfn(5, 8), 13, "Result B #1");
					a(i, 2, "Called B #1");
					a(mfn(3, 7), 10, "Result #3");
					a(i, 2, "Called #3");
					a(mfn(5, 8), 13, "Result B #2");
					a(i, 2, "Called B #2");

					setTimeout(function () {
						a(mfn(3, 7), 10, "Result: Wait");
						a(i, 2, "Called: Wait");
						a(mfn(5, 8), 13, "Result: Wait B");
						a(i, 2, "Called: Wait B");

						setTimeout(function () {
							a(mfn(3, 7), 10, "Result: Wait After");
							a(i, 3, "Called: Wait After");
							a(mfn(5, 8), 13, "Result: Wait After B");
							a(i, 4, "Called: Wait After B");

							a(mfn(3, 7), 10, "Result: Wait After #2");
							a(i, 4, "Called: Wait After #2");
							a(mfn(5, 8), 13, "Result: Wait After B #2");
							a(i, 4, "Called: Wait After B #2");
							d();
						}, 100);
					}, 20);
				},
				Async: function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, primitive: true, maxAge: 100 });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
					}), u, "Initial");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #1");
					}), u, "Initial #2");
					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #3");
					}), u, "Initial #2");
					a(mfn(5, 8, function (err, res) {
						a.deep([err, res], [null, 13], "Result B #2");
					}), u, "Initial #3");

					setTimeout(function () {
						a(i, 2, "Called #2");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Again: Result");
						}), u, "Again: Initial");
						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Again B: Result");
						}), u, "Again B: Initial");

						setTimeout(function () {
							a(i, 2, "Again Called #2");

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Again: Result");
							}), u, "Again: Initial");
							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Again B: Result");
							}), u, "Again B: Initial");

							nextTick(function () {
								a(i, 4, "Call After clear");
								d();
							});
						}, 100);
					}, 20);
				}
			}
		},
		Max: {
			Regular: {
				Sync: function (a) {
					var mfn, fn, i = 0;
					fn = function (x, y) {
						++i;
						return x + y;
					};
					mfn = t(fn, { max: 3 });

					a(mfn(3, 7), 10, "Result #1");
					a(i, 1, "Called #1");
					a(mfn(3, 7), 10, "Result #2");
					a(i, 1, "Called #2");
					a(mfn(5, 8), 13, "Result B #1");
					a(i, 2, "Called B #1");
					a(mfn(3, 7), 10, "Result #3");
					a(i, 2, "Called #3");
					a(mfn(5, 8), 13, "Result B #2");
					a(i, 2, "Called B #2");
					a(mfn(12, 4), 16, "Result C #1");
					a(i, 3, "Called C #1");
					a(mfn(3, 7), 10, "Result #4");
					a(i, 3, "Called #4");
					a(mfn(5, 8), 13, "Result B #3");
					a(i, 3, "Called B #3");

					a(mfn(77, 11), 88, "Result D #1"); // Clear 12, 4
					a(i, 4, "Called D #1");
					a(mfn(5, 8), 13, "Result B #4");
					a(i, 4, "Called B #4");
					a(mfn(12, 4), 16, "Result C #2"); // Clear 3, 7
					a(i, 5, "Called C #2");

					a(mfn(3, 7), 10, "Result #5"); // Clear 77, 11
					a(i, 6, "Called #5");
					a(mfn(77, 11), 88, "Result D #2"); // Clear 5, 8
					a(i, 7, "Called D #2");
					a(mfn(12, 4), 16, "Result C #3");
					a(i, 7, "Called C #3");

					a(mfn(5, 8), 13, "Result B #5"); // Clear 3, 7
					a(i, 8, "Called B #5");

					a(mfn(77, 11), 88, "Result D #3");
					a(i, 8, "Called D #3");

					mfn.delete(77, 11);
					a(mfn(77, 11), 88, "Result D #4");
					a(i, 9, "Called D #4");

					mfn.clear();
					a(mfn(5, 8), 13, "Result B #6");
					a(i, 10, "Called B #6");
					a(mfn(77, 11), 88, "Result D #5");
					a(i, 11, "Called D #5");
				},
				Async: function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, max: 3 });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
						a(i, 1, "Called #1");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Result #2");
							a(i, 1, "Called #2");

							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Result B #1");
								a(i, 2, "Called B #1");

								a(mfn(3, 7, function (err, res) {
									a.deep([err, res], [null, 10], "Result #3");
									a(i, 2, "Called #3");

									a(mfn(5, 8, function (err, res) {
										a.deep([err, res], [null, 13], "Result B #2");
										a(i, 2, "Called B #2");

										a(mfn(12, 4, function (err, res) {
											a.deep([err, res], [null, 16], "Result C #1");
											a(i, 3, "Called C #1");

											a(mfn(3, 7, function (err, res) {
												a.deep([err, res], [null, 10], "Result #4");
												a(i, 3, "Called #4");

												a(mfn(5, 8, function (err, res) {
													a.deep([err, res], [null, 13], "Result B #3");
													a(i, 3, "Called B #3");

													a(mfn(77, 11, function (err, res) {
														a.deep([err, res], [null, 88], "Result D #1");
														a(i, 4, "Called D #1");

														a(mfn(5, 8, function (err, res) {
															a.deep([err, res], [null, 13], "Result B #4");
															a(i, 4, "Called B #4");

															a(mfn(12, 4, function (err, res) {
																a.deep([err, res], [null, 16], "Result C #2");
																a(i, 5, "Called C #2");

																a(mfn(3, 7, function (err, res) {
																	a.deep([err, res], [null, 10], "Result #5");
																	a(i, 6, "Called #5");

																	a(mfn(77, 11, function (err, res) {
																		a.deep([err, res], [null, 88],
																			"Result D #2");
																		a(i, 7, "Called D #2");

																		a(mfn(12, 4, function (err, res) {
																			a.deep([err, res], [null, 16],
																				"Result C #3");
																			a(i, 7, "Called C #3");

																			a(mfn(5, 8, function (err, res) {
																				a.deep([err, res], [null, 13],
																					"Result B #5");
																				a(i, 8, "Called B #5");

																				a(mfn(77, 11, function (err, res) {
																					a.deep([err, res], [null, 88],
																						"Result D #3");
																					a(i, 8, "Called D #3");

																					mfn.delete(77, 11);
																					a(mfn(77, 11, function (err, res) {
																						a.deep([err, res], [null, 88],
																							"Result D #4");
																						a(i, 9, "Called D #4");

																						mfn.clear();
																						a(mfn(5, 8, function (err, res) {
																							a.deep([err, res], [null, 13],
																								"Result B #6");
																							a(i, 10, "Called B #6");

																							a(mfn(77, 11,
																								function (err, res) {
																									a.deep([err, res], [null, 88],
																										"Result D #5");
																									a(i, 11, "Called D #5");

																									d();
																								}), u, "Initial D #5");
																						}), u, "Initial B #6");
																					}), u, "Initial D #4");
																				}), u, "Initial D #3");
																			}), u, "Initial B #5");
																		}), u, "Initial C #3");
																	}), u, "Initial D #2");
																}), u, "Initial #5");
															}), u, "Initial C #2");
														}), u, "Initial B #4");
													}), u, "Initial D #1");
												}), u, "Initial B #3");
											}), u, "Initial #4");
										}), u, "Initial C #1");
									}), u, "Initial B #2");
								}), u, "Initial #3");
							}), u, "Initial B #1");
						}), u, "Initial #2");
					}), u, "Initial #1");
				}
			},
			Primitive: {
				Sync: function (a) {
					var mfn, fn, i = 0;
					fn = function (x, y) {
						++i;
						return x + y;
					};
					mfn = t(fn, { primitive: true, max: 3 });

					a(mfn(3, 7), 10, "Result #1");
					a(i, 1, "Called #1");
					a(mfn(3, 7), 10, "Result #2");
					a(i, 1, "Called #2");
					a(mfn(5, 8), 13, "Result B #1");
					a(i, 2, "Called B #1");
					a(mfn(3, 7), 10, "Result #3");
					a(i, 2, "Called #3");
					a(mfn(5, 8), 13, "Result B #2");
					a(i, 2, "Called B #2");
					a(mfn(12, 4), 16, "Result C #1");
					a(i, 3, "Called C #1");
					a(mfn(3, 7), 10, "Result #4");
					a(i, 3, "Called #4");
					a(mfn(5, 8), 13, "Result B #3");
					a(i, 3, "Called B #3");

					a(mfn(77, 11), 88, "Result D #1"); // Clear 12, 4
					a(i, 4, "Called D #1");
					a(mfn(5, 8), 13, "Result B #4");
					a(i, 4, "Called B #4");
					a(mfn(12, 4), 16, "Result C #2"); // Clear 3, 7
					a(i, 5, "Called C #2");

					a(mfn(3, 7), 10, "Result #5"); // Clear 77, 11
					a(i, 6, "Called #5");
					a(mfn(77, 11), 88, "Result D #2"); // Clear 5, 8
					a(i, 7, "Called D #2");
					a(mfn(12, 4), 16, "Result C #3");
					a(i, 7, "Called C #3");

					a(mfn(5, 8), 13, "Result B #5"); // Clear 3, 7
					a(i, 8, "Called B #5");

					a(mfn(77, 11), 88, "Result D #3");
					a(i, 8, "Called D #3");

					mfn.delete(77, 11);
					a(mfn(77, 11), 88, "Result D #4");
					a(i, 9, "Called D #4");

					mfn.clear();
					a(mfn(5, 8), 13, "Result B #6");
					a(i, 10, "Called B #6");
					a(mfn(77, 11), 88, "Result D #5");
					a(i, 11, "Called D #5");
				},
				Async: function (a, d) {
					var mfn, fn, u = {}, i = 0;
					fn = function (x, y, cb) {
						nextTick(function () {
							++i;
							cb(null, x + y);
						});
						return u;
					};

					mfn = t(fn, { async: true, primitive: true, max: 3 });

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #1");
						a(i, 1, "Called #1");

						a(mfn(3, 7, function (err, res) {
							a.deep([err, res], [null, 10], "Result #2");
							a(i, 1, "Called #2");

							a(mfn(5, 8, function (err, res) {
								a.deep([err, res], [null, 13], "Result B #1");
								a(i, 2, "Called B #1");

								a(mfn(3, 7, function (err, res) {
									a.deep([err, res], [null, 10], "Result #3");
									a(i, 2, "Called #3");

									a(mfn(5, 8, function (err, res) {
										a.deep([err, res], [null, 13], "Result B #2");
										a(i, 2, "Called B #2");

										a(mfn(12, 4, function (err, res) {
											a.deep([err, res], [null, 16], "Result C #1");
											a(i, 3, "Called C #1");

											a(mfn(3, 7, function (err, res) {
												a.deep([err, res], [null, 10], "Result #4");
												a(i, 3, "Called #4");

												a(mfn(5, 8, function (err, res) {
													a.deep([err, res], [null, 13], "Result B #3");
													a(i, 3, "Called B #3");

													a(mfn(77, 11, function (err, res) {
														a.deep([err, res], [null, 88], "Result D #1");
														a(i, 4, "Called D #1");

														a(mfn(5, 8, function (err, res) {
															a.deep([err, res], [null, 13], "Result B #4");
															a(i, 4, "Called B #4");

															a(mfn(12, 4, function (err, res) {
																a.deep([err, res], [null, 16], "Result C #2");
																a(i, 5, "Called C #2");

																a(mfn(3, 7, function (err, res) {
																	a.deep([err, res], [null, 10], "Result #5");
																	a(i, 6, "Called #5");

																	a(mfn(77, 11, function (err, res) {
																		a.deep([err, res], [null, 88],
																			"Result D #2");
																		a(i, 7, "Called D #2");

																		a(mfn(12, 4, function (err, res) {
																			a.deep([err, res], [null, 16],
																				"Result C #3");
																			a(i, 7, "Called C #3");

																			a(mfn(5, 8, function (err, res) {
																				a.deep([err, res], [null, 13],
																					"Result B #5");
																				a(i, 8, "Called B #5");

																				a(mfn(77, 11, function (err, res) {
																					a.deep([err, res], [null, 88],
																						"Result D #3");
																					a(i, 8, "Called D #3");

																					mfn.delete(77, 11);
																					a(mfn(77, 11, function (err, res) {
																						a.deep([err, res], [null, 88],
																							"Result D #4");
																						a(i, 9, "Called D #4");

																						mfn.clear();
																						a(mfn(5, 8, function (err, res) {
																							a.deep([err, res], [null, 13],
																								"Result B #6");
																							a(i, 10, "Called B #6");

																							a(mfn(77, 11,
																								function (err, res) {
																									a.deep([err, res], [null, 88],
																										"Result D #5");
																									a(i, 11, "Called D #5");

																									d();
																								}), u, "Initial D #5");
																						}), u, "Initial B #6");
																					}), u, "Initial D #4");
																				}), u, "Initial D #3");
																			}), u, "Initial B #5");
																		}), u, "Initial C #3");
																	}), u, "Initial D #2");
																}), u, "Initial #5");
															}), u, "Initial C #2");
														}), u, "Initial B #4");
													}), u, "Initial D #1");
												}), u, "Initial B #3");
											}), u, "Initial #4");
										}), u, "Initial C #1");
									}), u, "Initial B #2");
								}), u, "Initial #3");
							}), u, "Initial B #1");
						}), u, "Initial #2");
					}), u, "Initial #1");
				}
			}
		},
		Dispose: {
			Regular: {
				Sync: function (a) {
					var mfn, fn, value = [], x, invoked;
					fn = function (x, y) { return x + y; };
					mfn = t(fn, { dispose: function (val) { value.push(val); } });

					mfn(3, 7);
					mfn(5, 8);
					mfn(12, 4);
					a.deep(value, [], "Pre");
					mfn.delete(5, 8);
					a.deep(value, [13], "#1");
					value = [];
					mfn.delete(12, 4);
					a.deep(value, [16], "#2");

					value = [];
					mfn(77, 11);
					mfn.clear();
					a.deep(value, [10, 88], "Clear all");

					x = {};
					invoked = false;
					mfn = t(function () { return x; },
						{ dispose: function (val) { invoked = val; } });

					mfn.delete();
					a(invoked, false, "No args: Post invalid clear");
					mfn();
					a(invoked, false, "No args: Post cache");
					mfn.delete();
					a(invoked, x, "No args: Pre clear");
				},
				"Ref counter": function (a) {
					var mfn, fn, value = [];
					fn = function (x, y) { return x + y; };
					mfn = t(fn, { refCounter: true,
						dispose: function (val) { value.push(val); } });

					mfn(3, 7);
					mfn(5, 8);
					mfn(12, 4);
					a.deep(value, [], "Pre");
					mfn(5, 8);
					mfn.deleteRef(5, 8);
					a.deep(value, [], "Pre");
					mfn.deleteRef(5, 8);
					a.deep(value, [13], "#1");
					value = [];
					mfn.deleteRef(12, 4);
					a.deep(value, [16], "#2");

					value = [];
					mfn(77, 11);
					mfn.clear();
					a.deep(value, [10, 88], "Clear all");
				},
				Async: function (a, d) {
					var mfn, fn, u = {}, value = [];
					fn = function (x, y, cb) {
						nextTick(function () { cb(null, x + y); });
						return u;
					};

					mfn = t(fn, { async: true,
						dispose: function (val) { value.push(val); } });

					mfn(3, 7, function () {
						mfn(5, 8, function () {
							mfn(12, 4, function () {
								a.deep(value, [], "Pre");
								mfn.delete(5, 8);
								a.deep(value, [13], "#1");
								value = [];
								mfn.delete(12, 4);
								a.deep(value, [16], "#2");

								value = [];
								mfn(77, 11, function () {
									mfn.clear();
									a.deep(value, [10, 88], "Clear all");
									d();
								});
							});
						});
					});
				}
			},
			Primitive: {
				Sync: function (a) {
					var mfn, fn, value = [];
					fn = function (x, y) { return x + y; };
					mfn = t(fn, { dispose: function (val) { value.push(val); } });

					mfn(3, 7);
					mfn(5, 8);
					mfn(12, 4);
					a.deep(value, [], "Pre");
					mfn.delete(5, 8);
					a.deep(value, [13], "#1");
					value = [];
					mfn.delete(12, 4);
					a.deep(value, [16], "#2");

					value = [];
					mfn(77, 11);
					mfn.clear();
					a.deep(value, [10, 88], "Clear all");
				},
				"Ref counter": function (a) {
					var mfn, fn, value = [];
					fn = function (x, y) { return x + y; };
					mfn = t(fn, { refCounter: true,
						dispose: function (val) { value.push(val); } });

					mfn(3, 7);
					mfn(5, 8);
					mfn(12, 4);
					a.deep(value, [], "Pre");
					mfn(5, 8);
					mfn.deleteRef(5, 8);
					a.deep(value, [], "Pre");
					mfn.deleteRef(5, 8);
					a.deep(value, [13], "#1");
					value = [];
					mfn.deleteRef(12, 4);
					a.deep(value, [16], "#2");

					value = [];
					mfn(77, 11);
					mfn.clear();
					a.deep(value, [10, 88], "Clear all");
				},
				Async: function (a, d) {
					var mfn, fn, u = {}, value = [];
					fn = function (x, y, cb) {
						nextTick(function () { cb(null, x + y); });
						return u;
					};

					mfn = t(fn, { async: true,
						dispose: function (val) { value.push(val); } });

					mfn(3, 7, function () {
						mfn(5, 8, function () {
							mfn(12, 4, function () {
								a.deep(value, [], "Pre");
								mfn.delete(5, 8);
								a.deep(value, [13], "#1");
								value = [];
								mfn.delete(12, 4);
								a.deep(value, [16], "#2");

								value = [];
								mfn(77, 11, function () {
									mfn.clear();
									a.deep(value, [10, 88], "Clear all");
									d();
								});
							});
						});
					});
				}
			}
		}
	};
};
