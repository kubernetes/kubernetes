'use strict';

var memoize  = require('../..')
  , nextTick = require('next-tick');

module.exports = function () {
	return {
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

				mfn = memoize(fn, { async: true });

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
							a(i, 3, "Init  After delete");
							a(invoked, 9, "Cb After delete");
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

				mfn = memoize(fn, { async: true, dispose: a.never });

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

				mfn = memoize(fn, { async: true, primitive: true });

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
							a(i, 3, "Call After delete");
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

				mfn = memoize(fn, { async: true, primitive: true });

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
			},
			"Primitive null arg case": function (a, d) {
				var x = {}, mfn = memoize(function f(id, cb) { cb(null, x); }, {
					async: true,
					primitive: true
				});

				mfn(null, function (err, res) {
					a.deep([err, res], [null, x], "Args");
					d();
				});
			}
		},
		"Sync Clear": function (a, d) {
			var mfn, fn;
			fn = function (x, cb) {
				nextTick(function () {
					cb(null, x);
				});
			};

			mfn = memoize(fn, { async: true });
			mfn(1, function (err, i) {
				a(i, 1, "First");
			});
			mfn.clear();
			mfn(2, function (err, i) {
				a(i, 2, "Second");
				d();
			});
		},
		"Sync Clear: Primitive": function (a, d) {
			var mfn, fn;
			fn = function (x, cb) {
				nextTick(function () {
					cb(null, x);
				});
			};
			mfn = memoize(fn, { async: true, primitive: true });

			mfn(2, function (err, i) {
				a(i, 2, "First");
			});
			mfn(1, function (err, i) {
				a(i, 1, "Second");
				nextTick(d);
			});
			mfn.clear();
			mfn(2, function (err, i) {
				a(i, 2, "Third");
			});
		}
	};
};
