import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.data.quandl import cboe_rvx
from quantopian.pipeline.factors import AverageDollarVolume
# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600
# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 2.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 2.0 / TOTAL_POSITIONS
def initialize(context):
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)

def make_pipeline():

    returns = (-1) * Returns(window_length = 21)
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.total_scanned_messages],
        window_length=21,
    )

    ty = Fundamentals.total_yield.latest #-7.77
    wps = Fundamentals.working_capital_per_share.latest #-7.53
    epst = Fundamentals.tangible_book_value_per_share.latest
    gro = Fundamentals.growth_score.latest
    dollar_volume = AverageDollarVolume(window_length=63)

    universe = QTradableStocksUS() & (dollar_volume > 10**7)

    value_winsorized = value.winsorize(min_percentile=0.15, max_percentile=0.85)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.15, max_percentile=0.85)
    eps_wisorized = wps.winsorize(min_percentile=0.15, max_percentile=0.85)
    ty_winsorized = ty.winsorize(min_percentile=0.15, max_percentile=0.85)
    gro_winsorize = gro.winsorize(min_percentile=0.15, max_percentile=0.85)
    returns_winsorize = returns.winsorize(min_percentile=0.05, max_percentile=0.85)
   
    # Here we combine our winsorized factors, z-scoring them to equalize their influence
    combined_factor = (
        value_winsorized.zscore() *0.15
        + gro_winsorize.zscore() *0.25 
        + sentiment_score_winsorized.zscore()* 0.2
        + eps_wisorized.zscore() *0.15 
        + ty_winsorized.zscore() *0.15
        + returns_winsorize.zscore() *0.1
    )

    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)

    long_short_screen = (longs | shorts)

    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    
    
    return pipe

def before_trading_start(context, data):

    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    context.risk_loadings = algo.pipeline_output('risk_factors')

def record_vars(context, data):

    algo.record(num_positions=len(context.portfolio.positions))
    
def rebalance(context, data):

    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings

    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)

    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)

    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )