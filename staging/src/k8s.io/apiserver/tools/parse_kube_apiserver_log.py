#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parser for kubernetes API server logs."""


import argparse
import datetime
import numpy
import re


LOG_LINE_RE = re.compile(
    '^(?P<severity>[DIWEF])(?P<month>\d\d)(?P<day>\d\d) '
    '(?P<timestamp>\d\d:\d\d:\d\d\.\d+) +(?P<pid>\d+) '
    '(?P<sourcefile>[^:]+):(?P<sourceline>\d+)\] (?P<meat>.*)')

RESP_LINE_START_RE = re.compile(
    '^(?P<method>[a-zA-Z]+) (?P<uri>/[^ ]*): \((?P<latency>\d[^)]+)\) (?P<status>\d+|hijacked)(?P<rest>.*)$')

RESP_LINE_END_RE = re.compile(
    '(?P<extra_info>.*?)\s+\[(?P<user_agent>.+) (?P<remote_addr>\S+)\]$')

_SAVE_TEXT = True
_NOW = datetime.date.today()
_KEEP_FORMAT = '%Y-%m-%d %H:%M:%S'
_LATENCY_SPLIT_RE = re.compile('([^\d.]+)')
_ALL_METHODS = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE']


class State(object):
  START, MAIN, IN_LOG_ENTRY, IN_TRACE = range(4)


class KubeApiServerLogParserException(Exception):
  pass


class InvalidParameterException(KubeApiServerLogParserException):
  """Invalid parameter specified by the user."""


class InvalidInputException(KubeApiServerLogParserException):
  """Invalid input found in the log file."""


class UnrecognizedLine(InvalidInputException):
  pass


class InvalidResponseCodeString(InvalidInputException):
  pass


class Entry(object):
  def __init__(self, line):
    if _SAVE_TEXT:
      self._line = line

  def __str__(self):
    return self._line


class Junk(Entry):
  pass


class FlagComplaint(Junk):
  pass


class LogEntry(Entry):
  def __init__(self, line_match):
    super(LogEntry, self).__init__(line_match.group(0))
    ts = datetime.datetime.strptime(line_match.group('timestamp'), '%H:%M:%S.%f')
    self._timestamp = datetime.datetime(_NOW.year, int(line_match.group('month')), int(line_match.group('day')),
                                        ts.hour, ts.minute, ts.second, ts.microsecond)

  def timestamp(self):
    return self._timestamp


class NonResponseLog(LogEntry):
  pass


class Multiline(LogEntry):
  def __init__(self, line_match):
    super(Multiline, self).__init__(line_match)
    if _SAVE_TEXT:
      self._lines = [self._line]

  def append(self, line):
    if not _SAVE_TEXT:
      return
    if len(self._lines) == 1 and (not line or line.isspace()):
      return
    self._lines.append(line)

  def __str__(self):
    return '\n'.join(str(x) for x in self._lines)


class ResponseLog(Multiline):
  def __init__(self, line_match, resp_start_match, resp_end_match=None):
    super(ResponseLog, self).__init__(line_match)
    self._latency_sec = parse_latency(resp_start_match.group('latency'))
    self._method = resp_start_match.group('method')
    self._uri = resp_start_match.group('uri')
    self._status = resp_start_match.group('status')

  def latency_sec(self):
    return self._latency_sec

  def end(self, resp_end_match):
    self.append(resp_end_match.group(0))
    pass # TODO: save for later

  def method(self):
    return self._method

  def uri(self):
    return self._uri

  def status(self):
    return self._status


class Trace(Multiline):
  pass


def parse_latency(latency_str):
  total = 0.0
  units_and_numbers = list(reversed(_LATENCY_SPLIT_RE.split(latency_str)[:-1]))
  for unit, count in zip(units_and_numbers[0::2], units_and_numbers[1::2]):
    if unit == 's':
      multiplier = 1.0
    elif unit == 'm':
      multiplier = 60.0
    elif unit == 'ms':
      multiplier = 0.001
    elif unit == 'µs':
      multiplier = 0.000001
    else:
      raise InvalidInputException('Unknown unit %s in %s' % (unit, latency_str))
    total += multiplier * float(count)
  return total


def get_entries(f):
  state = State.START
  trace = None
  log = None
  for line in f:
    line = line.rstrip()

    if state == State.IN_TRACE:
      if line.startswith('Trace'):
        trace.append(line)
        continue
      else:
        yield trace
        trace = None
        state = State.MAIN

    if state == State.START and line.startswith('Flag '):
      yield FlagComplaint(line)
      continue
    elif state in (State.IN_LOG_ENTRY,):
      resp_end_match = RESP_LINE_END_RE.match(line)
      if resp_end_match:
        log.end(resp_end_match)
        yield log
        log = None
        state = State.MAIN
      else:
        log.append(line)
        state = State.IN_LOG_ENTRY

    elif state in (State.START, State.MAIN):
      if line.startswith('[restful]'):
        yield Junk(line)
        continue
      line_match = LOG_LINE_RE.match(line)
      if not line_match:
        raise UnrecognizedLine(line)

      meat = line_match.group('meat')
      if meat.startswith('Trace['):
        trace = Trace(line_match)
        state = State.IN_TRACE
        continue

      resp_start_match = RESP_LINE_START_RE.match(meat)
      if not resp_start_match:
        yield NonResponseLog(line_match)
        continue

      rest = resp_start_match.group('rest')
      resp_end_match = RESP_LINE_END_RE.match(rest)
      if resp_end_match:
        yield ResponseLog(line_match, resp_start_match, resp_end_match)
        state = State.MAIN
      else:
        log = ResponseLog(line_match, resp_start_match)
        log.append(rest)
        state = State.IN_LOG_ENTRY
      continue


def get_10ms(dt):
  rounded_usec = dt.microsecond - (dt.microsecond % 10000)
  return dt.replace(microsecond=rounded_usec)


def get_100ms(dt):
  rounded_usec = dt.microsecond - (dt.microsecond % 100000)
  return dt.replace(microsecond=rounded_usec)


def get_second(dt):
  return dt.replace(microsecond=0)


def get_10s(dt):
  rounded_sec = dt.second - (dt.second % 10)
  return dt.replace(second=rounded_sec, microsecond=0)


def get_minute(dt):
  return dt.replace(second=0, microsecond=0)


def get_10m(dt):
  rounded_min = dt.minute - (dt.minute % 10)
  return dt.replace(minute=rounded_min, second=0, microsecond=0)


def get_hour(dt):
  return dt.replace(minute=0, second=0, microsecond=0)


def get_all(dt):
  return datetime.datetime.utcfromtimestamp(0)


class Counter(object):
  HEADER = 'count'

  def __init__(self):
    self._count = 0

  def add(self, entry):
    self._count += 1

  def __str__(self):
    return str(self._count)


class ResponseCodeCounter(object):
  def __init__(self, pattern):
    self._pattern = pattern
    self._counter = Counter()

  def add(self, entry):
    if not hasattr(entry, 'status'):
      return
    status = entry.status()
    for index, value in self._pattern.iteritems():
      if status[index] != value:
        return
    self._counter.add(entry)

  def __str__(self):
    return str(self._counter)


def response_code_counter_provider(resp_code_str):
  if len(resp_code_str) != 3:
    raise InvalidResponseCodeString()
  pattern = dict()
  for i in xrange(3):
    if resp_code_str[i] in ('x', 'X'):
      pass
    elif resp_code_str[i].isdigit():
      pattern[i] = resp_code_str[i]
    else:
      raise InvalidResponseCodeString()
  getter = lambda: ResponseCodeCounter(pattern)
  getter.HEADER = resp_code_str + ' response count'
  return getter


class Latency(object):
  HEADER = 'mean latency'

  def __init__(self, percentile):
    self._data = []
    self._percentile = percentile

  def add(self, entry):
    self._data.append(entry.latency_sec())

  def __str__(self):
    return str(numpy.percentile(numpy.array(self._data), self._percentile))


def latency_percentile_getter(percentile):
  getter = lambda: Latency(percentile)
  getter.HEADER = '%d%%-ile latency' % percentile
  return getter


class TimestampFilter(object):
  def __init__(self, keep_after_str, keep_before_str):
    self._after = datetime.datetime.strptime(keep_after_str, _KEEP_FORMAT) if keep_after_str else None
    self._before = datetime.datetime.strptime(keep_before_str, _KEEP_FORMAT) if keep_before_str else None

  def should_discard(self, entry):
    if not hasattr(entry, 'timestamp'):
      return True
    if self._after and entry.timestamp() < self._after:
      return True
    if self._before and entry.timestamp() > self._before:
      return True
    return False


class MethodFilter(object):
  def __init__(self, methods_to_keep, methods_to_skip):
    self._discard_methodless = True if methods_to_keep else False
    self._keep = methods_to_keep if methods_to_keep else _ALL_METHODS
    self._skip = methods_to_skip if methods_to_skip else []

  def should_discard(self, entry):
    if not hasattr(entry, 'method'):
      return self._discard_methodless
    return entry.method() in self._skip or entry.method() not in self._keep


class WatchFilter(object):
  def should_discard(self, entry):
    if not hasattr(entry, 'uri'):
      return False
    return 'watch=true' in entry.uri()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--keep_after', help='Keep only records with timestamp after this (format: %s).' % _KEEP_FORMAT.replace('%', '%%'))
  parser.add_argument('--keep_before', help='Keep only records with timestamp before this (format: %s).' % _KEEP_FORMAT.replace('%', '%%'))
  parser.add_argument('--skip_method', action='append', choices=_ALL_METHODS,
                      help='Skip records with these methods.')
  parser.add_argument('--keep_method', action='append', choices=_ALL_METHODS,
                      help='Skip records with methods other than these.')
  parser.add_argument('--skip_watches', action='store_true',
                      help='Skip watch requests.')
  subparsers = parser.add_subparsers(dest='mode')
  parser.add_argument('file')

  aggregate = subparsers.add_parser('aggregate', help='Aggregate matching records and print CSV output.')
  aggregate.add_argument('--bucket_size', choices=['10ms', '100ms', 'second', '10s', 'minute', '10m', 'hour', 'all'], default='10s',
                         help='What bucket size to use.')
  aggregate.add_argument('--bucket_semantics', choices=['start', 'intersection', 'end'], default='end',
                         help='Add an event to the bucket if the start, duration or end, respectively - overlaps with the bucket.')
  aggregate.add_argument('--count_columns', choices=['all'], nargs=argparse.ONE_OR_MORE, default=[],
                         help='Which records to count.')
  aggregate.add_argument('--count_response_code_columns', nargs=argparse.ONE_OR_MORE, default=[],
                         help='Records with which response codes to count: values between 100 and 599, you can mask digits with "x", e.g. 5xx to count all server-side errors.')
  aggregate.add_argument('--latency_columns', type=int, nargs=argparse.ONE_OR_MORE, default=[],
                         help='Which response latency percentile to calculate and print: values between 0 and 100 expected.')

  copy = subparsers.add_parser('copy', help='Just copy matching records to output in text format.')

  args = parser.parse_args()

  timestamp_filter = TimestampFilter(args.keep_after, args.keep_before)
  method_filter = MethodFilter(args.keep_method, args.skip_method)
  watch_filter = WatchFilter() if args.skip_watches else None
  if args.mode == 'aggregate':
    if args.bucket_size == '10ms':
      bucketer = get_10ms
    elif args.bucket_size == '100ms':
      bucketer = get_100ms
    elif args.bucket_size == 'second':
      bucketer = get_second
    elif args.bucket_size == '10s':
      bucketer = get_10s
    elif args.bucket_size == 'minute':
      bucketer = get_minute
    elif args.bucket_size == '10m':
      bucketer = get_10m
    elif args.bucket_size == 'hour':
      bucketer = get_hour
    elif args.bucket_size == 'all':
      bucketer = get_all
    else:
      raise InvalidParameterException('Unsupported bucket size: ' + args.bucket_size)
    printer_factories = []
    for print_data in set(args.count_columns):
      if print_data == 'all':
        printer_factories.append(Counter)
      else:
        raise InvalidParameterException('Unsupported count column: ' + print_data)
    for print_data in set(args.count_response_code_columns):
      try:
        printer_factories.append(response_code_counter_provider(print_data))
      except InvalidResponseCodeString:
        raise InvalidParameterException('Unsupported response code string: ' + print_data)
    for print_data in set(args.latency_columns):
      if 0 <= print_data <= 100:
        printer_factories.append(latency_percentile_getter(print_data))
      else:
        raise InvalidParameterException('Unsupported latency column: ' + print_data)
    if not printer_factories:
      raise InvalidParameterException('No columns to print selected.')
    buckets = {}
  elif args.mode == 'copy':
    pass
  else:
    raise InvalidParameterException('Unknown mode ' + args.mode)

  with open(args.file, 'r') as f:
    for entry in get_entries(f):
      if timestamp_filter.should_discard(entry):
        continue
      if method_filter.should_discard(entry):
        continue
      if watch_filter and watch_filter.should_discard(entry):
        continue
      if args.mode == 'aggregate':
        if args.latency_columns and not hasattr(entry, 'latency_sec'):
          continue
        bucket = bucketer(entry.timestamp())
        if bucket not in buckets:
          buckets[bucket] = [printer_factory() for printer_factory in printer_factories]
        for aggregator in buckets[bucket]:
          aggregator.add(entry)
      elif args.mode == 'copy':
        print entry
  if args.mode == 'aggregate':
    print ';'.join([''] + [pf.HEADER for pf in printer_factories])
    for bucket in sorted(buckets):
      print ';'.join([str(bucket)] + [str(p) for p in buckets[bucket]])


if __name__ == '__main__':
  main()
