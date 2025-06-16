//Leetcode 1434
use std::collections::HashSet;

fn number_ways(curr_mask: usize, curr_hat: usize, hat_options: &Vec<HashSet<i32>>, all_hats: &HashSet<i32>, mut dp: &mut Vec<Vec<i64>>) -> i64{
    const MOD: i64 = 10_i64.pow(9) + 7;
    let size = hat_options.len();
    
    if curr_mask == (1<<size) - 1 {
        return 1
    }
    
    if curr_hat > 40 {
        return 0
    }
    
    if dp[curr_mask][curr_hat] != -1 {
        return dp[curr_mask][curr_hat]
    }
    
    let mut ans: i64 = number_ways(curr_mask, curr_hat + 1, &hat_options, &all_hats, &mut dp);
    
    if all_hats.contains(&(curr_hat as i32)) { 
        for j in 0..size {
            if curr_mask & (1<<j) == 0 && hat_options[j].contains(&(curr_hat as i32)) {
                ans += number_ways(curr_mask | (1<<j), curr_hat + 1, &hat_options, &all_hats, &mut dp);
                ans % MOD;
            }
        }
    }
    
    if ans > 0 {
        dp[curr_mask][curr_hat] = ans % MOD;
    } else {
        while ans < 0 {
            ans += MOD;
        }
        ans %= MOD;
        dp[curr_mask][curr_hat] = ans % MOD;
    }
    
    return dp[curr_mask][curr_hat]
    
}

impl Solution {
    pub fn number_ways(hats: Vec<Vec<i32>>) -> i32 {
        const MOD: i64 = 10_i64.pow(9) + 7;
        let curr_mask: usize = 0;
        let curr_hat: usize = 1;
        let size = hats.len();
        let mut all_hats: HashSet<i32> = HashSet::new();
        let mut hat_options: Vec<HashSet<i32>> = vec![HashSet::new(); size];
        for j in 0..size {
            let ind_size = hats[j].len();
            for i in 0..ind_size {
                all_hats.insert(hats[j][i]);
                hat_options[j].insert(hats[j][i]);
            }
        }
        let mut dp: Vec<Vec<i64>> = vec![vec![-1; 41];1<<size];
        let mut res = number_ways(curr_mask, curr_hat, &hat_options, &all_hats, &mut dp);

        if res > 0 {
            return (res % MOD) as i32
        } else {
            while res < 0{
                res += MOD;
            }
            return (res % MOD) as i32
        }

        
    }
}

//Leetcode 1542
use std::cmp;
use std::collections::HashMap;

impl Solution {
    pub fn longest_awesome(s: String) -> i32 {
        let str: Vec<char> = s.chars().collect();
        let arr: Vec<u32> = str.iter().map(|char| char.to_digit(10).unwrap()).collect(); 
    
        let size = str.len();
        let mut map: HashMap<i32, usize> = HashMap::new();
        let mut mask: i32 = 0;
        let mut ans: usize = 1; 
    
        map.insert(0, 0);
        for idx in 0..size {
            mask ^= 1<<(arr[idx]);
            if map.contains_key(&mask) {
                let prev_mask = *map.get(&mask).unwrap();
                if prev_mask == 0 {
                    ans = cmp::max(ans, idx + 1 - prev_mask);
                } else {
                    ans = cmp::max(ans, idx + 2 - prev_mask);
                } 
            }else {
                map.insert(mask, idx + 1);
            }
        
            for i in 0..10 {
                let prev_mask = mask ^ (1<<i);
                if i != arr[idx] {
                    if map.contains_key(&prev_mask) {
                        ans = cmp::max(ans, idx + 1 - *map.get(&prev_mask).unwrap());
                    }
                } else {
                    if mask & (mask - 1) == 0 {
                        ans = cmp::max(ans, idx + 1);
                    }
                }
            } 
        }
    
        return ans as i32
    }
}

//Leetcode 2035 (Inefficient) 
use std::cmp;
use std::collections::BTreeMap;

fn count_bits(num: &usize) -> i32 {
    let mut count: i32 = 0;
    let mut val = *num;
    
    while val > 0 {
        val = val &(val - 1);
        count += 1;
    }
    
    return count
}

impl Solution {
    pub fn minimum_difference(nums: Vec<i32>) -> i32 {
        let size = nums.len();
        let X: Vec<i32> = nums[..size/2].to_vec();
        let Y: Vec<i32> = nums[size/2..].to_vec();
        let mut dp1: Vec<Vec<i32>> = vec![vec![0,0]; 1<<(X.len())];
        let mut dp2: Vec<Vec<i32>> = vec![vec![0,0]; 1<<(Y.len())];
        let mut bit_count: Vec<BTreeMap<i32, i32>> = vec![BTreeMap::new(); size/2 + 2];
        bit_count[0].insert(0,1);
    
        for state in 0..(1<<(X.len())) as usize {
            for j in 0..X.len() {
                if state & 1<<j == 0 {
                    let count = count_bits(&state);
                    dp1[state | 1<<j] = vec![dp1[state][0] + X[j], 1 + count];
                }
            }
        }
    
        for state in 0..(1<<(Y.len())) as usize {
            for j in 0..Y.len() {
                if state & 1<<j == 0 {
                    let count = count_bits(&state);
                    dp2[state | 1<<j] = vec![dp2[state][0] + Y[j], 1 + count];
                    bit_count[count as usize + 1].insert(2*(dp2[state][0] + Y[j]),1);
                }
            }
        }
    
        let target = dp1[dp1.len() - 1][0] + dp2[dp2.len() - 1][0];
        let mut ans: i32 = i32::MAX;
        for i in 0..dp1.len() {
            let num_bits = dp1[i][1] as usize;
            if bit_count[size/2 - num_bits].is_empty() {
                continue
            } else {
                let next_val = bit_count[size/2 - num_bits].range(target - 2*dp1[i][0]..).next();
                let prev_val = bit_count[size/2 - num_bits].range(..target - 2*dp1[i][0]).next_back();
           
                if next_val.is_some() && prev_val.is_some() {
                    let cand1 = (*next_val.unwrap().0 + 2*dp1[i][0] - target).abs();
                    let cand2 = (*prev_val.unwrap().0 + 2*dp1[i][0] - target).abs();
                    ans = cmp::min(ans, cmp::min(cand1, cand2));
                } else if next_val.is_some() {
                    let cand1 = (*next_val.unwrap().0 + 2*dp1[i][0] - target).abs();
                    ans =  cmp::min(ans, cand1);
                } else {
                    let cand2 = (*prev_val.unwrap().0 + 2*dp1[i][0] - target).abs();
                    ans = cmp::min(ans, cand2);
                }
            }
       
            if ans == 0 {
                break 
            }
        }
   
        return ans
    }
}
