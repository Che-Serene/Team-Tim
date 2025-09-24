// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interfaces:srv/GetPose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "interfaces/srv/get_pose.hpp"


#ifndef INTERFACES__SRV__DETAIL__GET_POSE__TRAITS_HPP_
#define INTERFACES__SRV__DETAIL__GET_POSE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interfaces/srv/detail/get_pose__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetPose_Request & msg,
  std::ostream & out)
{
  (void)msg;
  out << "null";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetPose_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetPose_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace interfaces

namespace rosidl_generator_traits
{

[[deprecated("use interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interfaces::srv::GetPose_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const interfaces::srv::GetPose_Request & msg)
{
  return interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interfaces::srv::GetPose_Request>()
{
  return "interfaces::srv::GetPose_Request";
}

template<>
inline const char * name<interfaces::srv::GetPose_Request>()
{
  return "interfaces/srv/GetPose_Request";
}

template<>
struct has_fixed_size<interfaces::srv::GetPose_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<interfaces::srv::GetPose_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<interfaces::srv::GetPose_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetPose_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: position
  {
    if (msg.position.size() == 0) {
      out << "position: []";
    } else {
      out << "position: [";
      size_t pending_items = msg.position.size();
      for (auto item : msg.position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: orientation
  {
    if (msg.orientation.size() == 0) {
      out << "orientation: []";
    } else {
      out << "orientation: [";
      size_t pending_items = msg.orientation.size();
      for (auto item : msg.orientation) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetPose_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.position.size() == 0) {
      out << "position: []\n";
    } else {
      out << "position:\n";
      for (auto item : msg.position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: orientation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.orientation.size() == 0) {
      out << "orientation: []\n";
    } else {
      out << "orientation:\n";
      for (auto item : msg.orientation) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetPose_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace interfaces

namespace rosidl_generator_traits
{

[[deprecated("use interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interfaces::srv::GetPose_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const interfaces::srv::GetPose_Response & msg)
{
  return interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interfaces::srv::GetPose_Response>()
{
  return "interfaces::srv::GetPose_Response";
}

template<>
inline const char * name<interfaces::srv::GetPose_Response>()
{
  return "interfaces/srv/GetPose_Response";
}

template<>
struct has_fixed_size<interfaces::srv::GetPose_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interfaces::srv::GetPose_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interfaces::srv::GetPose_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetPose_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetPose_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetPose_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace interfaces

namespace rosidl_generator_traits
{

[[deprecated("use interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interfaces::srv::GetPose_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const interfaces::srv::GetPose_Event & msg)
{
  return interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interfaces::srv::GetPose_Event>()
{
  return "interfaces::srv::GetPose_Event";
}

template<>
inline const char * name<interfaces::srv::GetPose_Event>()
{
  return "interfaces/srv/GetPose_Event";
}

template<>
struct has_fixed_size<interfaces::srv::GetPose_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interfaces::srv::GetPose_Event>
  : std::integral_constant<bool, has_bounded_size<interfaces::srv::GetPose_Request>::value && has_bounded_size<interfaces::srv::GetPose_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<interfaces::srv::GetPose_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<interfaces::srv::GetPose>()
{
  return "interfaces::srv::GetPose";
}

template<>
inline const char * name<interfaces::srv::GetPose>()
{
  return "interfaces/srv/GetPose";
}

template<>
struct has_fixed_size<interfaces::srv::GetPose>
  : std::integral_constant<
    bool,
    has_fixed_size<interfaces::srv::GetPose_Request>::value &&
    has_fixed_size<interfaces::srv::GetPose_Response>::value
  >
{
};

template<>
struct has_bounded_size<interfaces::srv::GetPose>
  : std::integral_constant<
    bool,
    has_bounded_size<interfaces::srv::GetPose_Request>::value &&
    has_bounded_size<interfaces::srv::GetPose_Response>::value
  >
{
};

template<>
struct is_service<interfaces::srv::GetPose>
  : std::true_type
{
};

template<>
struct is_service_request<interfaces::srv::GetPose_Request>
  : std::true_type
{
};

template<>
struct is_service_response<interfaces::srv::GetPose_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // INTERFACES__SRV__DETAIL__GET_POSE__TRAITS_HPP_
