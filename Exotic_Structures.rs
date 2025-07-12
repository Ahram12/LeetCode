//Leetcode 3605
//Sparse Table

use std::cmp;

fn gcd(a: &i32, b: &i32) -> i32 {
    let mut p: i32 = cmp::min(*a, *b);
    let mut q: i32 = cmp::max(*a, *b);
    
    if p == 0 {
        return q
    }
    
    loop {
        if q % p == 0 {
            return p
        } else {
            let temp = p;
            q %= p;
            p = q;
            q = temp;
        }
    }
}


fn main() {
    let arr: Vec<i32> = vec![2, 6, 8];
    let size = arr.len();
    let max_c: usize = 2;
    
    let mut max_len: usize = 0;
    while 1<<max_len < size {
        max_len += 1;
    }
    
    let mut sparse_table: Vec<Vec<i32>> = vec![vec![0; size]; max_len + 1];
    
    for i in 0..size {
        sparse_table[0][i] = arr[i]
    }
    
    for i in 1..=max_len {
        let mut j: usize = 0;
        while j + (1<< (i - 1))  < size {
            sparse_table[i][j] = gcd(&sparse_table[i - 1][j], &sparse_table[i - 1][j + (1<<(i - 1))]);
            j += 1;
        }
    }
    
    let mut log: Vec<usize> = vec![0; size + 1];
    for i in 2..=size {
        log[i] = log[i/2] + 1;
    }
    
    let mut low: usize = 1;
    let mut high: usize = size;
    let mut max_len: usize = size + 1;
    
    while low <= high {
        let mid = (low + high)/2;
        let mut temp_c: usize = max_c;
        let mut j: usize = mid - 1;
        let mut flag: bool = true;
        
        while j < size {
            let l: usize = j + 1 - mid;
            let r: usize = j;
            let i: usize = log[mid];
            if gcd(&sparse_table[i][l], &sparse_table[i][r + 1 - (1<<i)]) == 1 {
                j += 1;
            } else {
                if temp_c > 0 {
                    temp_c -= 1;
                    j += mid;
                } else {
                    flag = false;
                    break
                }
            }
        }
        
        if flag {
            max_len = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
        
    }
    
    
    println!("{:?}", max_len - 1);
}
