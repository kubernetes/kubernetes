'use strict';

var memoize  = require('../..')
  , nextTick = require('next-tick');

require('../../ext/async');

module.exports = function () {
	return {
		Regular: {
			Sync: function (a, d) {
				var mfn, fn, i = 0;
				fn = function (x, y) {
					++i;
					return x + y;
				};
				mfn = memoize(fn, { maxAge: 100 });

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

					a(mfn(9, 1), 10, "Result: C");
					a(i, 3, "Called: C");
					setTimeout(function () {
						a(mfn(3, 7), 10, "Result: Wait After");
						a(i, 4, "Called: Wait After");
						a(mfn(5, 8), 13, "Result: Wait After B");
						a(i, 5, "Called: Wait After B");

						a(mfn(3, 7), 10, "Result: Wait After #2");
						a(i, 5, "Called: Wait After #2");
						a(mfn(5, 8), 13, "Result: Wait After B #2");
						a(i, 5, "Called: Wait After B #2");

						a(mfn(9, 1), 10, "Result: WiatC");
						a(i, 5, "Called: Wait C");
						d();
					}, 90);
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

				mfn = memoize(fn, { async: true, maxAge: 100 });

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
				mfn = memoize(fn, { primitive: true, maxAge: 100 });

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

				mfn = memoize(fn, { async: true, primitive: true, maxAge: 100 });

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
		Refetch: {
			Default: function (a, d) {
				var mfn, fn, i = 0;
				fn = function (x, y) {
					++i;
					return x + y;
				};
				mfn = memoize(fn, { maxAge: 600, preFetch: true });

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
						a(i, 2, "Called: Wait After");
						a(mfn(5, 8), 13, "Result: Wait After B");
						a(i, 2, "Called: Wait After B");

						a(mfn(3, 7), 10, "Result: Wait After #2");
						a(i, 2, "Called: Wait After #2");
						a(mfn(5, 8), 13, "Result: Wait After B #2");
						a(i, 2, "Called: Wait After B #2");

						setTimeout(function () {
							a(i, 4, "Called: After Refetch: Before");
							a(mfn(3, 7), 10, "Result: After Refetch");
							a(i, 4, "Called: After Refetch: After");
							a(mfn(5, 8), 13, "Result: After Refetch B");
							a(i, 4, "Called: After Refetch B: After");

							setTimeout(function () {
								a(mfn(3, 7), 10, "Result: After Refetch #2");
								a(i, 4, "Called: After Refetch #2");
								a(mfn(5, 8), 13, "Result: After Refetch #2 B");
								a(i, 4, "Called: After Refetch #2 B");

								a(mfn(3, 7), 10, "Result: After Refetch #3");
								a(i, 4, "Called: After Refetch #3");
								a(mfn(5, 8), 13, "Result: After Refetch #3 B");
								a(i, 4, "Called: After Refetch #3 B");
								d();
							}, 200);
						}, 200);

					}, 200);
				}, 300);
			},
			Async: function (a, d) {
				var mfn, fn, i = 0;
				fn = function (x, y, cb) {
					++i;
					setTimeout(function () { cb(null, x + y); }, 0);
				};
				mfn = memoize(fn, { maxAge: 600, preFetch: true, async: true });

				mfn(3, 7, function (err, result) {
					a(result, 10, "Result #1");
					a(i, 1, "Called #1");
					mfn(3, 7, function (err, result) {
						a(result, 10, "Result #2");
						a(i, 1, "Called #2");
						mfn(5, 8, function (err, result) {
							a(result, 13, "Result B #1");
							a(i, 2, "Called B #1");
							mfn(3, 7, function (err, result) {
								a(result, 10, "Result #3");
								a(i, 2, "Called #3");
								mfn(5, 8, function (err, result) {
									a(result, 13, "Result B #2");
									a(i, 2, "Called B #2");
									setTimeout(function () {
										mfn(3, 7, function (err, result) {
											a(result, 10, "Result: Wait");
											a(i, 2, "Called: Wait");
											mfn(5, 8, function (err, result) {
												a(result, 13, "Result: Wait B");
												a(i, 2, "Called: Wait B");
												setTimeout(function () {
													mfn(3, 7, function (err, result) {
														a(result, 10, "Result: Wait After");
														a(i, 2, "Called: Wait After");
														mfn(5, 8, function (err, result) {
															a(result, 13, "Result: Wait After B");
															a(i, 2, "Called: Wait After B");
															mfn(3, 7, function (err, result) {
																a(result, 10, "Result: Wait After #2");
																a(i, 2, "Called: Wait After #2");
																mfn(5, 8, function (err, result) {
																	a(result, 13, "Result: Wait After B #2");
																	a(i, 2, "Called: Wait After B #2");
																	setTimeout(function () {
																		a(i, 4, "Called: After Refetch: Before");
																		mfn(3, 7, function (err, result) {
																			a(result, 10, "Result: After Refetch");
																			a(i, 4, "Called: After Refetch: After");
																			mfn(5, 8, function (err, result) {
																				a(result, 13, "Result: After Refetch B");
																				a(i, 4, "Called: After Refetch B: After");
																				setTimeout(function () {
																					mfn(3, 7, function (err, result) {
																						a(result, 10, "Result: After Refetch #2");
																						a(i, 4, "Called: After Refetch #2");
																						mfn(5, 8, function (err, result) {
																							a(result, 13, "Result: After Refetch #2 B");
																							a(i, 4, "Called: After Refetch #2 B");
																							mfn(3, 7, function (err, result) {
																								a(result, 10, "Result: After Refetch #3");
																								a(i, 4, "Called: After Refetch #3");
																								mfn(5, 8, function (err, result) {
																									a(result, 13, "Result: After Refetch #3 B");
																									a(i, 4, "Called: After Refetch #3 B");
																									d();
																								});
																							});
																						});
																					});
																				}, 200);
																			});
																		});
																	}, 200);

																});
															});
														});
													});
												}, 200);
											});
										});
									}, 300);
								});
							});
						});
					});
				});

			},
			Custom: function (a, d) {
				var mfn, fn, i = 0;
				fn = function (x, y) {
					++i;
					return x + y;
				};
				mfn = memoize(fn, { maxAge: 600, preFetch: 1 / 6 });
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
						a(i, 2, "Called: Wait After");
						a(mfn(5, 8), 13, "Result: Wait After B");
						a(i, 2, "Called: Wait After B");

						a(mfn(3, 7), 10, "Result: Wait After #2");
						a(i, 2, "Called: Wait After #2");
						a(mfn(5, 8), 13, "Result: Wait After B #2");
						a(i, 2, "Called: Wait After B #2");

						setTimeout(function () {
							a(i, 4, "Called: After Refetch: Before");
							a(mfn(3, 7), 10, "Result: After Refetch");
							a(i, 4, "Called: After Refetch: After");
							a(mfn(5, 8), 13, "Result: After Refetch B");
							a(i, 4, "Called: After Refetch B: After");

							setTimeout(function () {
								a(mfn(3, 7), 10, "Result: After Refetch #2");
								a(i, 4, "Called: After Refetch #2");
								a(mfn(5, 8), 13, "Result: After Refetch #2 B");
								a(i, 4, "Called: After Refetch #2 B");

								a(mfn(3, 7), 10, "Result: After Refetch #3");
								a(i, 4, "Called: After Refetch #3");
								a(mfn(5, 8), 13, "Result: After Refetch #3 B");
								a(i, 4, "Called: After Refetch #3 B");
								d();
							}, 200);
						}, 300);

					}, 100);
				}, 450);
			}
		}
	};
};
