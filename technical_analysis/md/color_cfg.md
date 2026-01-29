重要：读图时优先获取颜色设置，严格按照下面颜色设置读图
// ===================================
// 🎨 Color Preset (From Original Files + DMI+DD v6)
// ===================================

// —— ADX / DMI (adx_dd.pine) ——
color C_ADX        = color.red
color C_DI_PLUS    = color.blue
color C_DI_MINUS   = color.gray
color C_ADX_LEVEL  = color.white   // Key Level for ADX

// —— KDJ + RSI (kdjrsi.pine) ——
color C_KD_AVG     = color.orange
color C_J          = color.fuchsia
color C_RSI_STRONG = color.purple
color C_RSI_WEAK   = color(purple, 80)

// —— BB (bb_chips_cost.pine) ——
color C_BB_BASIS_UP    = color.lime
color C_BB_BASIS_DOWN  = color.red
color C_BB_UPPER       = #F23645
color C_BB_LOWER       = #089981
color C_BB_FILL        = color.rgb(33, 150, 243, 95)

// —— Chips Cost (bb_chips_cost.pine) ——
color C_CHIPS_COST     = color.new(color.purple, 70)

// —— ATR Stop Loss (atr_sl.pine / integrated) ——
// 上止（Short Stop）→ 洋红
// 下止（Long Stop） → 绿色
color C_ATR_STOP_HIGH  = color.new(color.fuchsia, 70)
color C_ATR_STOP_LOW   = color.new(color.green, 70)

// —— Drawdown / Max Drawdown (adx_dd.pine) ——
color C_DRAWDOWN       = color.orange
color C_MAX_DRAWDOWN   = color.new(color.purple, 70)

// —— MACD (macd_pro.pine) ——
color C_MACD_UP        = color.lime
color C_MACD_DOWN      = color.red
color C_SIGNAL         = color.yellow

color C_HIST_UP_A      = color.aqua
color C_HIST_UP_B      = color.maroon
color C_HIST_DOWN_A    = color.blue
color C_HIST_DOWN_B    = color.red

color C_MACD_ZERO_LINE = color.white