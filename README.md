# Candlestick-Data-Training-with-LSTM

This is script for training LSTM model for candlestick data prediction.
There are 3 different model structures in this script.

1. Vanilla:
    This is simple structure containing 3 layers. 
    1st layer is LSTM with 150 units.
    2nd layer is Dense layer which extract output info from 1st layer.
    3rd layer is again dense layer for getting final result that is Close value.
    It uses mae as loss function.
 
2. Biderectional:
    It uses bidrectional layers infused with LSTM layers.
    There are total 3 bidirectional layers infused with LSTM.
    It uses dense layer for extraction of LSTM features.
    It uses same mae as loss function.
    
3. Stack:-
    It is same as vanilla model just with extra model layers.
    
You can train and save all models with any candlestick timeframe ( 1m, 2m, 5m etc )
