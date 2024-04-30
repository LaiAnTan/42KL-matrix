from typing import List


def col(mat, idx):
    
    return [mat[row][idx] for row in range(len(mat))]

def row_echelon(mat: List[List[float]]) -> List[List[float]]:

    rows = len(mat)
    cols = len(mat[0])

    # find first non zero column
    first_non_zero_col_idx = 0
    for i in range(cols):
        
        if sum(col(mat, i)) == 0:
            first_non_zero_col_idx = i
            break
    
    # find pivot index (row) in first non zero column (first non zero element)
    pivot_idx = 0
    for i in range(rows):
        
        if col(mat, first_non_zero_col_idx)[i] != 0:
            pivot_idx = i
            break
    
    for i in range(first_non_zero_col_idx, cols):
        
        if pivot_idx == rows:
            break
        
        curr_col = col(mat, i)

        # pivot is 0, special logic
        if mat[pivot_idx][i] == 0:

            # if all elem below pivot in the curr col are 0 we do nothing and go to the next column
            if sum(col(mat, pivot_idx)[pivot_idx + 1:]) == 0:
                continue
            
            # else find first non zero row below pivot row
            for j in range(pivot_idx + 1, rows):
                
                # swap with pivot row
                if mat[j][pivot_idx] != 0:
                    mat[pivot_idx], mat[j] = mat[j], mat[pivot_idx]
        
        # scale pivot row
        mat[pivot_idx] = [item / mat[pivot_idx][i] for item in mat[pivot_idx]]
        
        # handle rows after pivot (make 0)
        for k in range(pivot_idx + 1, rows):
            
            # make current col 0 if not yet 0
            if curr_col[k] != 0:
                mat[k] = [mat[k][x] - (curr_col[k] * mat[pivot_idx][x]) for x in range(cols)]
                
                
        pivot_idx += 1
    
    return mat

def reduced_echelon(mat: List[List[float]]) -> List[List[float]]:
    
    mat = row_echelon(mat)
    
    rows = len(mat)
    cols = len(mat[0])
    
    # start from last row until second last row (last row pivot does not do anything)
    for i in range((rows - 1), 0, -1):

        # find 1 in row (col index)
        for k in range((cols - 1), -1, -1):
            
            if mat[i][k] == 1:
                pivot_idx = k
                break
        else:
            continue
        
        # if everything above already 0, we go to the next pivot in the row above
        if sum(col(mat, pivot_idx)[0:i]) == 0:
            continue
        
        # make everything above 0 using pivot row
        # row = row - (pivot_row * scalar)
        for j in range((i - 1), -1, -1):
            
            if mat[j][pivot_idx] != 0:
                mat[j] = [mat[j][x] - (mat[j][pivot_idx] * mat[i][x]) for x in range(cols)]
    
    return mat


if __name__ == "__main__":
    
    # test no pivot column
    mat = [
        [8, 5, -2, 4, 28],
        [4, 2.5, 20, 4, -4],
        [8, 5, 1, 4, 17]
        ]
    
    # test swap rows
    # mat = [
    #     [8, 4],
    #     [4, 2],
    #     [1, 5],
    #     ]
    
    # test full zero row
    # mat = [
    #     [1, 2],
    #     [2, 4]
    # ]
    
    # mat = [
    #     [1, 2, -1],
    #     [2, 5, -1],
    #     [0, -1, 4]
    # ]
    
    print(f"REF: {row_echelon(mat)}")
    print(f"RREF: {reduced_echelon(mat)}")


