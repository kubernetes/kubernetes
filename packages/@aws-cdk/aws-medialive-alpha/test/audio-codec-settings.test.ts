import { Bitrate } from 'aws-cdk-lib';
import {
  AudioCodecSettings,
  AudioCodecType,
  AacProfile,
  AacCodingMode,
  AacRateControlMode,
  AacRawFormat,
  AacSpec,
  AacInputType,
  AacVbrQuality,
  Ac3CodingMode,
  Ac3AttenuationControl,
  Ac3BitstreamMode,
  Ac3DrcProfile,
  Ac3LfeFilter,
  Ac3MetadataControl,
  Eac3CodingMode,
  Eac3DrcLine,
  Eac3DrcRf,
  Eac3StereoDownmix,
  Eac3AtmosCodingMode,
  Eac3AtmosDrcLine,
  Mp2CodingMode,
  WavCodingMode,
  AudioSampleRate,
  AudioBitDepth,
} from '../lib';

describe('AAC', () => {
  test('applies defaults', () => {
    const settings = AudioCodecSettings.aac();
    expect(settings._codecType).toBe(AudioCodecType.AAC);
    expect(settings._bind()).toEqual({
      aacSettings: {
        bitrate: 192000,
        profile: 'LC',
        codingMode: 'CODING_MODE_2_0',
        rateControlMode: 'CBR',
        sampleRate: 48000,
        rawFormat: 'NONE',
        spec: 'MPEG4',
        inputType: 'NORMAL',
      },
    });
  });

  test('passes explicit values', () => {
    const settings = AudioCodecSettings.aac({
      bitrate: Bitrate.kbps(256),
      profile: AacProfile.HEV1,
      codingMode: AacCodingMode.CODING_MODE_5_1,
      rateControlMode: AacRateControlMode.VBR,
      sampleRate: AudioSampleRate.HZ_44100,
      rawFormat: AacRawFormat.LATM_LOAS,
      spec: AacSpec.MPEG2,
      inputType: AacInputType.BROADCASTER_MIXED_AD,
      vbrQuality: AacVbrQuality.HIGH,
    });
    expect(settings._bind()).toEqual({
      aacSettings: {
        bitrate: 256000,
        profile: 'HEV1',
        codingMode: 'CODING_MODE_5_1',
        rateControlMode: 'VBR',
        sampleRate: 44100,
        rawFormat: 'LATM_LOAS',
        spec: 'MPEG2',
        inputType: 'BROADCASTER_MIXED_AD',
        vbrQuality: 'HIGH',
      },
    });
  });
});

describe('AC3', () => {
  test('applies console defaults', () => {
    const settings = AudioCodecSettings.ac3();
    expect(settings._codecType).toBe(AudioCodecType.AC3);
    expect(settings._bind()).toEqual({
      ac3Settings: {
        codingMode: 'CODING_MODE_2_0',
        bitstreamMode: 'COMPLETE_MAIN',
        lfeFilter: 'DISABLED',
      },
    });
  });

  test('passes explicit values and maps dialNorm to dialnorm', () => {
    const settings = AudioCodecSettings.ac3({
      bitrate: Bitrate.kbps(384),
      codingMode: Ac3CodingMode.CODING_MODE_3_2_LFE,
      dialNorm: 24,
      attenuationControl: Ac3AttenuationControl.ATTENUATE_3_DB,
      bitstreamMode: Ac3BitstreamMode.COMPLETE_MAIN,
      drcProfile: Ac3DrcProfile.FILM_STANDARD,
      lfeFilter: Ac3LfeFilter.ENABLED,
      metadataControl: Ac3MetadataControl.USE_CONFIGURED,
    });
    expect(settings._bind()).toEqual({
      ac3Settings: {
        bitrate: 384000,
        codingMode: 'CODING_MODE_3_2_LFE',
        dialnorm: 24,
        attenuationControl: 'ATTENUATE_3_DB',
        bitstreamMode: 'COMPLETE_MAIN',
        drcProfile: 'FILM_STANDARD',
        lfeFilter: 'ENABLED',
        metadataControl: 'USE_CONFIGURED',
      },
    });
  });
});

describe('EAC3', () => {
  test('applies console defaults', () => {
    const settings = AudioCodecSettings.eac3();
    expect(settings._codecType).toBe(AudioCodecType.EAC3);
    expect(settings._bind()).toEqual({
      eac3Settings: {
        codingMode: 'CODING_MODE_3_2',
        attenuationControl: 'NONE',
        bitstreamMode: 'COMPLETE_MAIN',
      },
    });
  });

  test('passes explicit values', () => {
    const settings = AudioCodecSettings.eac3({
      bitrate: Bitrate.kbps(384),
      codingMode: Eac3CodingMode.CODING_MODE_3_2,
      dialNorm: 20,
      drcLine: Eac3DrcLine.FILM_STANDARD,
      drcRf: Eac3DrcRf.MUSIC_LIGHT,
      loRoCenterMixLevel: -3,
      stereoDownmix: Eac3StereoDownmix.LO_RO,
    });
    expect(settings._bind()).toEqual({
      eac3Settings: {
        bitrate: 384000,
        codingMode: 'CODING_MODE_3_2',
        dialnorm: 20,
        attenuationControl: 'NONE',
        bitstreamMode: 'COMPLETE_MAIN',
        drcLine: 'FILM_STANDARD',
        drcRf: 'MUSIC_LIGHT',
        loRoCenterMixLevel: -3,
        stereoDownmix: 'LO_RO',
      },
    });
  });
});

describe('EAC3 Atmos', () => {
  test('applies console defaults', () => {
    const settings = AudioCodecSettings.eac3Atmos();
    expect(settings._codecType).toBe(AudioCodecType.EAC3_ATMOS);
    expect(settings._bind()).toEqual({
      eac3AtmosSettings: {
        codingMode: 'CODING_MODE_5_1_4',
      },
    });
  });

  test('passes explicit values', () => {
    const settings = AudioCodecSettings.eac3Atmos({
      bitrate: Bitrate.kbps(768),
      codingMode: Eac3AtmosCodingMode.CODING_MODE_7_1_4,
      dialNorm: 18,
      drcLine: Eac3AtmosDrcLine.FILM_LIGHT,
      heightTrim: 1,
      surroundTrim: 2,
    });
    expect(settings._bind()).toEqual({
      eac3AtmosSettings: {
        bitrate: 768000,
        codingMode: 'CODING_MODE_7_1_4',
        dialnorm: 18,
        drcLine: 'FILM_LIGHT',
        heightTrim: 1,
        surroundTrim: 2,
      },
    });
  });
});

describe('MP2', () => {
  test('applies console defaults', () => {
    const settings = AudioCodecSettings.mp2();
    expect(settings._codecType).toBe(AudioCodecType.MP2);
    expect(settings._bind()).toEqual({
      mp2Settings: { codingMode: 'CODING_MODE_2_0', sampleRate: 48000 },
    });
  });

  test('passes explicit values', () => {
    const settings = AudioCodecSettings.mp2({
      bitrate: Bitrate.kbps(128),
      codingMode: Mp2CodingMode.CODING_MODE_2_0,
      sampleRate: AudioSampleRate.HZ_32000,
    });
    expect(settings._bind()).toEqual({
      mp2Settings: { bitrate: 128000, codingMode: 'CODING_MODE_2_0', sampleRate: 32000 },
    });
  });
});

describe('WAV', () => {
  test('applies console defaults', () => {
    const settings = AudioCodecSettings.wav();
    expect(settings._codecType).toBe(AudioCodecType.WAV);
    expect(settings._bind()).toEqual({
      wavSettings: { bitDepth: 16, codingMode: 'CODING_MODE_2_0', sampleRate: 48000 },
    });
  });

  test('passes explicit values', () => {
    const settings = AudioCodecSettings.wav({
      bitDepth: AudioBitDepth.DEPTH_24,
      codingMode: WavCodingMode.CODING_MODE_8_0,
      sampleRate: AudioSampleRate.HZ_96000,
    });
    expect(settings._bind()).toEqual({
      wavSettings: { bitDepth: 24, codingMode: 'CODING_MODE_8_0', sampleRate: 96000 },
    });
  });
});

describe('Passthrough', () => {
  test('produces empty passthrough settings', () => {
    const settings = AudioCodecSettings.passthrough();
    expect(settings._codecType).toBe(AudioCodecType.PASSTHROUGH);
    expect(settings._bind()).toEqual({ passThroughSettings: {} });
  });
});
