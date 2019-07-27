

def apply_transaction_cost(ret, close, slippage, security_type):
    base_rets = ret.copy()

    if security_type == "etf_stock":
        result_rets = base_rets
    elif security_type == "etf_etc":
        'ETF(기타): Tax = 거래이익 * 15.4%, 슬리피지 = 틱당 5원'
        result_rets = base_rets

        # 슬리피지 차감
        slip_rates = slippage * 5 / close
        slip_costs = slip_rates.loc[base_rets.index]
        result_rets = result_rets - slip_costs

        # 세금 차감
        tax_costs = base_rets[lambda x: x > 0] * 0.15
        tax_rets = (result_rets - tax_costs).dropna()
        result_rets.loc[tax_rets.index] = tax_rets


    return result_rets