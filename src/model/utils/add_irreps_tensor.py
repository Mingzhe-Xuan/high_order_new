from e3nn.o3 import Irreps
import torch
from typing import Union


def get_intersection_irreps(irreps1: Union[str, Irreps], irreps2: Union[str, Irreps]) -> Irreps:
    """
    Get the intersection of two irreps, taking the minimum multiplicity for each (l, p).
    """
    if isinstance(irreps1, str):
        irreps1 = Irreps(irreps1)
    if isinstance(irreps2, str):
        irreps2 = Irreps(irreps2)
    
    irreps1 = irreps1.sort()[0].simplify()
    irreps2 = irreps2.sort()[0].simplify()
    
    mul_dict1 = {ir: mul for mul, ir in irreps1}
    mul_dict2 = {ir: mul for mul, ir in irreps2}
    
    intersection = []
    for ir in mul_dict1:
        if ir in mul_dict2:
            intersection.append((min(mul_dict1[ir], mul_dict2[ir]), ir))
    
    return Irreps(intersection)


def selective_residual_add(
    irreps_in: Union[str, Irreps],
    irreps_out: Union[str, Irreps],
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Add residual connection only for common irrep components between irreps_in and irreps_out.
    
    For common (l, p) components: output = input + transformed_output
    For components only in irreps_out: output = transformed_output
    
    Returns a tensor with shape matching irreps_out.dim
    """
    if isinstance(irreps_in, str):
        irreps_in = Irreps(irreps_in)
    if isinstance(irreps_out, str):
        irreps_out = Irreps(irreps_out)
    
    irreps_in = irreps_in.sort()[0].simplify()
    irreps_out = irreps_out.sort()[0].simplify()
    
    batch_size = output_tensor.shape[0]
    device = output_tensor.device
    dtype = output_tensor.dtype
    
    result = torch.zeros(batch_size, irreps_out.dim, device=device, dtype=dtype)
    
    mul_dict_in = {ir: mul for mul, ir in irreps_in}
    slices_in = list(irreps_in.slices())
    slices_out = list(irreps_out.slices())
    
    in_offset = 0
    for i, (mul_out, ir_out) in enumerate(irreps_out):
        slice_out = slices_out[i]
        dim_out = mul_out * (2 * ir_out.l + 1)
        
        if ir_out in mul_dict_in:
            mul_in = mul_dict_in[ir_out]
            dim_in = mul_in * (2 * ir_out.l + 1)
            
            in_idx = None
            cumsum = 0
            for j, (mul_j, ir_j) in enumerate(irreps_in):
                if ir_j == ir_out:
                    in_idx = j
                    break
                cumsum += mul_j * (2 * ir_j.l + 1)
            
            if in_idx is not None:
                slice_in = slices_in[in_idx]
                min_mul = min(mul_in, mul_out)
                min_dim = min_mul * (2 * ir_out.l + 1)
                
                result[:, slice_out.start:slice_out.start + min_dim] = (
                    input_tensor[:, slice_in.start:slice_in.start + min_dim] +
                    output_tensor[:, slice_out.start:slice_out.start + min_dim]
                )
                
                if mul_out > mul_in:
                    extra_dim = (mul_out - mul_in) * (2 * ir_out.l + 1)
                    result[:, slice_out.start + min_dim:slice_out.stop] = (
                        output_tensor[:, slice_out.start + min_dim:slice_out.stop]
                    )
        else:
            result[:, slice_out.start:slice_out.stop] = output_tensor[:, slice_out.start:slice_out.stop]
    
    return result


def get_union_irreps(irreps_list: Union[list[Irreps], list[str]]) -> Irreps:
    # Note that irreps is a tuple of (mul, (l, p)), which is a data type named mul_ir.
    mul_dict = {}
    for irreps in irreps_list:
        if isinstance(irreps, str):
            irreps = Irreps(irreps)
        irreps = irreps.sort()[0].simplify()
        for mul, ir in irreps:
            mul_dict[ir] = max(mul_dict.get(ir, 0), mul)

    return Irreps([(mul_dict[l], l) for l in mul_dict.keys()])


def add_irreps_tensor(
    irreps_list: Union[list[Irreps], list[str]], tensor_list: list[torch.Tensor]
) -> torch.Tensor:
    assert len(irreps_list) == len(
        tensor_list
    ), "Length of irreps_list and tensor_list should be the same."
    # get the union irreps that contains all the ls in irreps_list and has the highest mul for each l
    union_irreps = get_union_irreps(irreps_list)
    # get index for each l in union_irreps
    union_ls = [l for mul, (l, p) in union_irreps]
    union_l_index = {l: i for i, l in enumerate(union_ls)}
    # slices for different ls
    union_slice_list = union_irreps.slices()
    batch_size = tensor_list[0].shape[0]
    for tensor in tensor_list:
        assert (
            tensor.shape[0] == batch_size
        ), "Batch size of tensors in tensor_list should be the same."

    # add tensors
    tensor_out = torch.zeros(
        (batch_size, union_irreps.dim),
        dtype=tensor_list[0].dtype,
        device=tensor_list[0].device,
    )
    for irreps, tensor in zip(irreps_list, tensor_list):
        if isinstance(irreps, str):
            irreps = Irreps(irreps)
        irreps = irreps.sort()[0].simplify()
        ls = [l for mul, (l, p) in irreps]
        l_index = {l: i for i, l in enumerate(ls)}
        slice_list = irreps.slices()
        sliced_tensor = [tensor[:, s.start : s.stop] for s in slice_list]
        padding = torch.zeros(
            (batch_size, union_irreps.dim),
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )
        for i, mul_ir in enumerate(irreps):
            l = mul_ir[1][0]  # (mul, (l, p))
            start = union_slice_list[union_l_index[l]].start
            end = start + mul_ir.dim
            padding[:, start:end] = sliced_tensor[l_index[l]]
        tensor_out = tensor_out + padding
    return tensor_out


# Also usable

# def add_irreps_tensor(
#     irreps_list: Union[list[Irreps], list[str]], tensor_list: list[torch.Tensor]
# ) -> torch.Tensor:
#     # get the union irreps that contains all the ls in irreps_list and has the highest mul for each l
#     union_irreps = get_union_irreps(irreps_list)
#     # add tensors
#     batch_size = tensor_list[0].shape[0]
#     for tensor in tensor_list:
#         assert tensor.shape[0] == batch_size, "Batch size of tensors in tensor_list should be the same."
#     tensor_out = torch.zeros(
#         (batch_size, union_irreps.dim), dtype=tensor_list[0].dtype, device=tensor_list[0].device
#     )

#     # Create mapping from (l, p) -> index in union irreps for easier lookup
#     union_irreps_map = {}  # Maps (l, p) to its index in union_irreps
#     union_cumsum = [0]  # Cumulative sum of dimensions in union irreps
#     for i, (mul, (l, p)) in enumerate(union_irreps):
#         union_irreps_map[(l, p)] = i
#         union_cumsum.append(union_cumsum[-1] + mul * (2*l + 1))

#     for irreps, tensor in zip(irreps_list, tensor_list):
#         if isinstance(irreps, str):
#             irreps = Irreps(irreps)
#         irreps = irreps.sort()[0].simplify()

#         # Process each component of the current irreps
#         current_cumsum = [0]  # Cumulative sum of dimensions in current irreps
#         for mul, (l, p) in irreps:
#             current_cumsum.append(current_cumsum[-1] + mul * (2*l + 1))

#         # Reset cumulative index for current iteration
#         current_idx = 0
#         union_idx = 0
#         for i, (mul, (l, p)) in enumerate(irreps):
#             # Get the tensor slice for this component
#             start_idx = current_cumsum[i]
#             end_idx = current_cumsum[i + 1]
#             component_tensor = tensor[:, start_idx:end_idx]  # Include batch dimension

#             # Find where this (l, p) component goes in the union
#             union_component_idx = union_irreps_map[(l, p)]
#             union_start = union_cumsum[union_component_idx]
#             union_end = union_cumsum[union_component_idx + 1]

#             # Create temporary tensor for this component with the right shape
#             temp_union_tensor = torch.zeros((batch_size, union_end - union_start),
#                                           dtype=tensor.dtype, device=tensor.device)

#             # Calculate how many copies of this irrep type we have in current vs union
#             current_dim_per_irrep = 2*l + 1  # dimension of single irrep
#             current_mult = mul  # multiplicity in current tensor
#             union_mult = union_irreps[union_component_idx][0]  # multiplicity in union

#             # Fill the appropriate part of temp tensor with component_tensor
#             # Map from current mult copies to union mult copies
#             for m in range(current_mult):
#                 start_pos = m * current_dim_per_irrep
#                 end_pos = (m + 1) * current_dim_per_irrep
#                 union_start_pos = m * current_dim_per_irrep
#                 union_end_pos = (m + 1) * current_dim_per_irrep
#                 temp_union_tensor[:, union_start_pos:union_end_pos] = \
#                     component_tensor[:, start_pos:end_pos]

#             # Add to the final tensor
#             tensor_out[:, union_start:union_end] += temp_union_tensor

#             current_idx = end_idx

#     return tensor_out
