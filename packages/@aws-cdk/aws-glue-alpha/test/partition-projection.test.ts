import * as glue from '../lib';

describe('PartitionProjectionConfiguration Validation', () => {
  describe('INTEGER validation', () => {
    test.each([
      [1.5, 10],
      [1, 10.5],
    ])('throws when min=%p or max=%p is not an integer', (min, max) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.integer({ min, max });
      }).toThrow(`INTEGER partition projection range must contain integers, but got [${min}, ${max}]`);
    });

    test('throws when min > max', () => {
      expect(() => {
        glue.PartitionProjectionConfiguration.integer({
          min: 10,
          max: 5,
        });
      }).toThrow('INTEGER partition projection range must be [min, max] where min <= max, but got [10, 5]');
    });

    test.each([0, -1, 1.5])('throws when interval=%p is invalid', (interval) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.integer({
          min: 1,
          max: 10,
          interval,
        });
      }).toThrow(`INTEGER partition projection interval must be a positive integer, but got ${interval}`);
    });

    test.each([0, -1, 1.5])('throws when digits=%p is invalid', (digits) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.integer({
          min: 1,
          max: 10,
          digits,
        });
      }).toThrow(`INTEGER partition projection digits must be an integer >= 1, but got ${digits}`);
    });
  });

  describe('DATE validation', () => {
    test.each([
      ['', '2023-12-31'],
      ['   ', '2023-12-31'],
      ['2020-01-01', ''],
    ])('throws when min=%p or max=%p is empty', (min, max) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({ min, max, format: 'yyyy-MM-dd' });
      }).toThrow('DATE partition projection range must not contain empty strings');
    });

    test.each(['', '   '])('throws when format=%p is empty', (format) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({
          min: '2020-01-01',
          max: '2023-12-31',
          format,
        });
      }).toThrow('DATE partition projection format must be a non-empty string');
    });

    test.each([
      'yyyy-MM-dd',
      'yyyy/MM/dd/HH',
      "yyyyMMdd'T'HHmmss",
      "yyyy-MM-dd''HH",
    ])('accepts valid format=%p', (format) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({
          min: '2020-01-01',
          max: '2023-12-31',
          format,
          // interval/intervalUnit supplied so the finer-than-day formats are valid;
          // this case exercises format-character acceptance, not the interval rule.
          interval: 1,
          intervalUnit: glue.DateIntervalUnit.HOURS,
        });
      }).not.toThrow();
    });

    // Sub-day precision (a field finer than a day) requires interval + unit.
    // `a` (AM/PM) counts as sub-day.
    test.each([
      'yyyy-MM-dd-HH', // hourly
      "yyyyMMdd'T'HHmmss", // to the second
      'yyyy-MM-dd a', // AM/PM — two partitions per day
    ])('requires interval and intervalUnit when format=%p has sub-day precision', (format) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({ min: '2020-01-01', max: '2023-12-31', format });
      }).toThrow(/has sub-day precision, so both 'interval' and 'intervalUnit' are required/);
    });

    // Day precision or coarser (month, year, quarter) does not require them —
    // Athena defaults the step.
    test.each([
      'yyyy-MM-dd', // day
      'yyyy-MM', // month
      'yyyy', // year (coarser than a month, yet still optional)
      'yyyy-QQ', // quarter
    ])('allows omitting interval/intervalUnit when format=%p is day precision or coarser', (format) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({ min: '2020', max: '2023', format });
      }).not.toThrow();
    });

    test('accepts a finer-than-day format when interval and intervalUnit are provided', () => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({
          min: '2020-01-01-00',
          max: '2023-12-31-23',
          format: 'yyyy-MM-dd-HH',
          interval: 1,
          intervalUnit: glue.DateIntervalUnit.HOURS,
        });
      }).not.toThrow();
    });

    test.each([
      ['yyyy-bb-dd', ['b']],
      ['yyyy-MM-ddJ', ['J']],
    ])('throws when format=%p contains invalid characters %p', (format, invalidChars) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({
          min: '2020-01-01',
          max: '2023-12-31',
          format,
        });
      }).toThrow(`DATE partition projection format contains invalid pattern characters: ${invalidChars.join(', ')}. Must use Java DateTimeFormatter valid pattern letters.`);
    });

    test('throws when format has unclosed single quote', () => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({
          min: '2020-01-01',
          max: '2023-12-31',
          format: "yyyy-MM-dd'T",
        });
      }).toThrow("DATE partition projection format has an unclosed single quote: 'yyyy-MM-dd'T'");
    });

    test.each([0, -1, 1.5])('throws when interval=%p is invalid', (interval) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.date({
          min: '2020-01-01',
          max: '2023-12-31',
          format: 'yyyy-MM-dd',
          interval,
        });
      }).toThrow(`DATE partition projection interval must be a positive integer, but got ${interval}`);
    });
  });

  describe('ENUM validation', () => {
    test('throws when values is empty array', () => {
      expect(() => {
        glue.PartitionProjectionConfiguration.enum({
          values: [],
        });
      }).toThrow('ENUM partition projection values must be a non-empty array');
    });

    test.each([
      [['us-east-1', '', 'us-west-2']],
      [['us-east-1', '   ', 'us-west-2']],
    ])('throws when values=%p contains empty string', (values) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.enum({ values });
      }).toThrow('ENUM partition projection values must not contain empty strings');
    });

    test.each([
      [['value,with,commas', 'normal'], 0],
      [['normal', 'also,bad'], 1],
    ])('throws when values=%p contains comma at index %p', (values, index) => {
      expect(() => {
        glue.PartitionProjectionConfiguration.enum({ values });
      }).toThrow(`ENUM partition projection values must not contain commas because the values are serialized as a comma-separated list, got: '${values[index]}'`);
    });
  });
});
