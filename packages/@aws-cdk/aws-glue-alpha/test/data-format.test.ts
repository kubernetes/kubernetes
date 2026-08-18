import { InputFormat, OutputFormat } from '../lib';

// Regression test: OutputFormat.AVRO and OutputFormat.ORC were previously
// constructed as `InputFormat` instances. That compiled in TypeScript because
// the two classes are structurally identical, but in jsii's nominally-typed
// languages (Java, C#, Go, Python) it made these members unusable wherever an
// `OutputFormat` is required. At runtime the classes have distinct prototypes,
// so `instanceof` distinguishes the bug (would fail below) from the fix.
describe('OutputFormat static members are OutputFormat instances', () => {
  test.each([
    ['AVRO', OutputFormat.AVRO, 'org.apache.hadoop.hive.ql.io.avro.AvroContainerOutputFormat'],
    ['ORC', OutputFormat.ORC, 'org.apache.hadoop.hive.ql.io.orc.OrcOutputFormat'],
    ['HIVE_IGNORE_KEY_TEXT', OutputFormat.HIVE_IGNORE_KEY_TEXT, 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'],
    ['PARQUET', OutputFormat.PARQUET, 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'],
  ])('OutputFormat.%s is an OutputFormat with an unchanged class name', (_name, format, className) => {
    expect(format).toBeInstanceOf(OutputFormat);
    expect(format).not.toBeInstanceOf(InputFormat);
    expect(format.className).toEqual(className);
  });
});
