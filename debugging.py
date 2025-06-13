import pandas as pd
import matplotlib.pyplot as plt

truth_df = pd.read_parquet("trident_data/truth_minimal.parquet")
cumulative = truth_df["cumulative_lengths"].values
lengths = cumulative[1:] - cumulative[:-1]
lengths = [cumulative[0]] + list(lengths) 

plt.hist(lengths, bins=50)
plt.xlabel("Hits per Event")
plt.ylabel("Number of Events")
plt.title("Distribution of Hits per Event")
plt.grid(True)
plt.show()