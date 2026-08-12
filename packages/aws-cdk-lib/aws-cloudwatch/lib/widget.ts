import type { IMetric } from './metric-types';
import { UnscopedValidationError } from '../../core';
import { lit } from '../../core/lib/private/literal-string';

/**
 * The width of the grid we're filling
 */
export const GRID_WIDTH = 24;

/**
 * A single dashboard widget
 */
export interface IWidget {
  /**
   * The amount of horizontal grid units the widget will take up
   */
  readonly width: number;

  /**
   * The amount of vertical grid units the widget will take up
   */
  readonly height: number;

  /**
   * Any warnings that are produced as a result of putting together this widget
   * @deprecated - use warningsV2
   */
  readonly warnings?: string[];

  /**
   * Any warnings that are produced as a result of putting together this widget
   */
  readonly warningsV2?: { [id: string]: string };

  /**
   * Place the widget at a given position
   */
  position(x: number, y: number): void;

  /**
   * Return the widget JSON for use in the dashboard
   */
  toJson(): any[];
}

/**
 * A real CloudWatch widget that has its own fixed size and remembers its position
 *
 * This is in contrast to other widgets which exist for layout purposes.
 */
export abstract class ConcreteWidget implements IWidget {
  public readonly width: number;
  public readonly height: number;
  protected x?: number;
  protected y?: number;

  public readonly warnings?: string[] = [];
  public readonly warningsV2?: { [id: string]: string } = {};

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;

    if (this.width > GRID_WIDTH) {
      throw new UnscopedValidationError(lit`WidgetTooWide`, `Widget is too wide, max ${GRID_WIDTH} units allowed`);
    }
  }

  public position(x: number, y: number): void {
    this.x = x;
    this.y = y;
  }

  public abstract toJson(): any[];

  /**
   * Copy the warnings from the given metric
   */
  protected copyMetricWarnings(...ms: IMetric[]) {
    ms.forEach(m => {
      for (const [id, message] of Object.entries(m.warningsV2 ?? {})) {
        this.warningsV2![id] = message;
      }
    });
  }
}
